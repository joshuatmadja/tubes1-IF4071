import org.w3c.dom.Attr;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.classifiers.AbstractClassifier;
import weka.core.Capabilities.Capability;
import weka.classifiers.evaluation.Evaluation;

import java.util.Enumeration;
import java.util.Vector;

public class MyNewC45  extends AbstractClassifier {
    private static final double EPSILON = 1e-6;
    MyNewC45[] nodes;
    Attribute attribute;
    Attribute class_attribute;
    double[] class_distribution;
    double class_value;
    double[] information_gains;
    double[] gain_ratios;
    Attribute[] attributes;
    static Vector<double[]> rules;
    static Vector<Double> accuracies;

    @Override
    public void buildClassifier(Instances instances) throws Exception {
        getCapabilities().testWithFail(instances);

        instances = new Instances(instances);
        instances.deleteWithMissingClass();

        attributes = new Attribute[instances.numAttributes() - 1];
        for (int i = 0; i < attributes.length; ++i) {
            attributes[i] = instances.attribute(i);
        }
        makeTree(instances);
        makeRules();
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        if (instance.hasMissingValue()) {

        }

        double result = 0.0;

        for (double[] rule: rules) {
            for (int i = 0; i < rule.length-1; ++i) {
                if (rule[i] != -1) {
                    if (rule[i] != instance.value(i)) {
                        break;
                    }
                }

                if (i == rule.length-2) {
                    result = rule[rule.length-1];
                }
            }
        }

        return result;
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        if (instance.hasMissingValue()) {

        }

        if (attribute == null) {
            return class_distribution;
        } else {
            return nodes[(int) instance.value(attribute)].distributionForInstance(instance);
        }
    }

    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();

        // attributes
        result.enable(Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capability.NUMERIC_ATTRIBUTES);

        // class
        result.enable(Capability.NOMINAL_CLASS);
        result.enable(Capability.MISSING_CLASS_VALUES);

        // instances
        result.setMinimumNumberInstances(0);

        return result;
    }

    public void makeTree(Instances data) throws Exception {
        attribute = null;
        class_distribution = getClassDistribution(data);

        if ((attributes.length == 0) || (areAllExamplesHaveSameClassValue(class_distribution))) {
            class_distribution = normalize(class_distribution);
            class_value = getMaxIndex(class_distribution);
            class_attribute = data.classAttribute();
        } else {
//            information_gains = new double[attributes.length];
//            for (int i = 0; i < attributes.length; ++i) {
//                information_gains[i] = getInformationGain(data, attributes[i]);
//            }
//            attribute = attributes[getMaxIndex(information_gains)];

            gain_ratios = new double[data.numAttributes()];
            for (int i = 0; i < attributes.length; ++i) {
                gain_ratios[i] = getGainRatio(data, attributes[i]);
            }
            attribute = attributes[getMaxIndex(gain_ratios)];

            Instances[] splits = getSplittedData(data, attribute);
            nodes = new MyNewC45[attribute.numValues()];

            for (int i = 0; i < attribute.numValues(); ++i) {
                nodes[i] = new MyNewC45();
                if (splits[i].isEmpty()) {
                    nodes[i].attribute = null;
                    nodes[i].class_distribution = class_distribution;
                    nodes[i].class_distribution = nodes[i].normalize(nodes[i].class_distribution);
                    nodes[i].class_value = nodes[i].getMaxIndex(nodes[i].class_distribution);
                    nodes[i].class_attribute = data.classAttribute();
                } else {
                    Attribute[] child_atts = new Attribute[attributes.length-1];

                    int k = 0;
                    for (int j = 0; j < attributes.length; ++j) {
                        if (attributes[j].index() != attribute.index()) {
                            child_atts[k] = attributes[j];
                            ++k;
                        }
                    }

                    nodes[i].attributes = child_atts;
                    nodes[i].makeTree(splits[i]);
                }
            }
        }

    }

    public double log2(double value) throws Exception {
        return Math.log(value) / Math.log(2);
    }

    public boolean isEqual(double a, double b) throws Exception {
        return (a - b < EPSILON) && (b - a < EPSILON);
    }

    public double[] normalize(double[] ds) {
        double[] result = ds;

        double sum = 0.0;
        for (double d : ds) {
            sum += d;
        }

        for (int i = 0; i < ds.length; ++i) {
            result[i] /= sum;
        }
        return result;
    }

    public int getMaxIndex(double[] array) throws Exception {
        double max = array[0];
        int idx = 0;

        for (int i = 1; i < array.length; ++i) {
            if (array[i] > max) {
                idx = i;
                max = array[i];
            }
        }

        return idx;
    }

    public boolean areAllExamplesHaveSameClassValue(double[] ds) throws Exception {
        int zero = 0;

        for (double d : ds) {
            if (d == 0) {
                ++zero;
            }
        }

        return (zero == ds.length - 1);
    }

    public Instances[] getSplittedData(Instances data, Attribute att) throws Exception {
        Instances[] splits;

        splits = new Instances[att.numValues()];

        for (int i = 0; i < att.numValues(); ++i) {
            splits[i] = new Instances(data, data.numInstances());
        }

        Enumeration<Instance> enumInst = data.enumerateInstances();
        while (enumInst.hasMoreElements()) {
            Instance in = enumInst.nextElement();
            splits[((int) in.value(att))].add(in);
        }

        for (int i = 0; i < splits.length; ++i) {
            splits[i].compactify();
        }

        return splits;
    }

    public double getEntropy(Instances data) throws Exception {
        double entropy = 0.0;
        double[] count_class_values = new double[data.numClasses()];

        Enumeration<Instance> enumInst = data.enumerateInstances();
        while (enumInst.hasMoreElements()) {
            Instance in = enumInst.nextElement();
            count_class_values[((int) in.classValue())]++;
        }

        for (int i = 0; i < data.numClasses(); ++i) {
            if (count_class_values[i] > 0) {
                double probability = count_class_values[i]/data.numInstances();
                entropy -= probability * log2(probability);
            }
        }

        return entropy;
    }

    public double getInformationGain(Instances data, Attribute att) throws Exception {
        double information_gain = getEntropy(data);
        Instances[] splits = getSplittedData(data, att);

        for (int i = 0; i < splits.length - 1; ++i) {
            if (splits[i].numInstances() > 0) {
                information_gain -= ((double) splits[i].numInstances() / (double) data.numInstances()) * getEntropy(splits[i]);
            }
        }

        return information_gain;
    }

    public double[] getClassDistribution(Instances data) throws Exception{
        double [] result = new double[data.numClasses()];

        Enumeration<Instance> enumInst = data.enumerateInstances();
        while (enumInst.hasMoreElements()) {
            Instance in = enumInst.nextElement();
            result[((int) in.classValue())]++;
        }

        return result;
    }

    private void makeRulesRecursive(double[] rule) throws Exception {
        if (attribute == null) {
            double[] new_rule = rule;
            new_rule[new_rule.length-1] = class_value;
            rules.add(new_rule);
        } else {
            for (int i = 0; i < nodes.length; ++i) {

                double[] new_rule = rule;
                new_rule[attribute.index()] = ((double) i);
                nodes[i].makeRulesRecursive(new_rule);
            }
        }
    }

    private void makeRules() throws Exception {
        rules = new Vector<>();
        for (int i = 0; i < nodes.length; ++i) {
            double[] rule = new double[attributes.length + 1];
            for (int j = 0; j < rule.length; ++j) {
                rule[j] = -1;
            }
            rule[attribute.index()] = ((double) i);
            nodes[i].makeRulesRecursive(rule);
        }

//        Vector<double[]> new_rules = new Vector<>();
//
//        for (int i = 0; i < rules.size(); ++i) {
//            if (new_rules.isEmpty()) {
//                new_rules.add(rules.get(i));
//            } else {
//                for (int j = 0; j < new_rules.size(); ++j) {
//                    for (int k = 0; k < new_rules.get(j).length; ++k) {
//                        if (isEqual(rules.get(j)[k], rules.get(i)[k])) {
//                            new_rules.add(rules.get(i));
//                        }
//                    }
//                }
//            }
//        }
//
//        rules = new_rules;

        for (double[] rule : rules) {
            for (double d : rule) {
                System.out.println(d);
            }
            System.out.println();
        }

        accuracies = new Vector<>(rules.size());
    }

    private double getSplitInformation(Instances data, Attribute att) throws Exception {
        double split_information = 0.0;
        Instances[] splits = getSplittedData(data, att);

        for (Instances split: splits) {
            if (split.numInstances() > 0) {
                double peluang = (double) split.numInstances() / (double) data.numInstances();
                split_information -= peluang * log2(peluang);
            }
        }

        return split_information;
    }

    private double getGainRatio(Instances data, Attribute att) throws Exception {
        double information_gain = getInformationGain(data, att);
        double split_information = getSplitInformation(data, att);
        double gain_ratio = information_gain / split_information;

        return gain_ratio;
    }
}
