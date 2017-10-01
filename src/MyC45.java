import org.w3c.dom.Attr;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.classifiers.AbstractClassifier;
import weka.core.Capabilities.Capability;

import java.util.Enumeration;
import java.util.Vector;

public class MyC45 extends AbstractClassifier {
    private static final double EPSILON = 1e-6;
    private MyC45[] nodes;
    private double[] information_gains;
    private double[] gain_ratios;
    private double[] classDistribution;
    private double classValue;
    private Attribute attribute;
    private Attribute classAttribute;
    private double[] att_thresholds;
    private static Vector<double[]> rules = new Vector<>();

    private void makeRulesRecursive(double[] rule) throws Exception {
        if (attribute == null) {
            rules.add(rule);
        } else {
            for (int i = 0; i < nodes.length; ++i) {

                double[] new_rule = rule;
                new_rule[attribute.index()] = ((double) i);
                nodes[i].makeRulesRecursive(new_rule);
            }
        }
    }

    private void makeRules() throws Exception {
        for (int i = 0; i < nodes.length; ++i) {
            double[] rule = new double[10];
            for (int j = 0; j < rule.length; ++j) {
                rule[j] = -1;
            }
            rule[attribute.index()] = ((double) i);
            nodes[i].makeRulesRecursive(rule);
        }
    }

    private double getMostCommonValue(Instances data, Attribute att, Instance in) throws Exception {
        Instances data_label = new Instances(data, data.numInstances());

        Enumeration<Instance> enumInst = data.enumerateInstances();
        while (enumInst.hasMoreElements()) {
            Instance in_1 = enumInst.nextElement();
            if ((in.classValue() == in_1.classValue()) && (!in_1.isMissing(att.index()))) {
                data_label.add(in_1);
            }
        }
        data_label.compactify();

        double[] count_values = new double[att.numValues()];
        for (int i = 0; i < data_label.numInstances(); ++i) {
            count_values[((int) data_label.instance(i).value(att))]++;
        }

        return ((double) getMaxIndex(count_values));
    }

    private Instances getDataWithoutMissingValues(Instances data) throws Exception {
        Instances result = new Instances(data, data.numInstances());

        Enumeration<Instance> enumInst = data.enumerateInstances();
        while (enumInst.hasMoreElements()){
            Instance in = enumInst.nextElement();
            if (in.hasMissingValue()){
                for (int i = 0; i < in.numAttributes(); ++i) {
                    if (in.isMissing(i)){
                        if (in.attribute(i).isNominal()) {
                            in.setValue(i, getMostCommonValue(data, in.attribute(i), in));
                        } else if (in.attribute(i).isNumeric()){
                            int sum = 0;
                            for (int j = 0; j < data.numInstances(); ++j){
                                if ((in.classValue() == data.instance(j).classValue()) && (!data.instance(j).isMissing(i))) {
                                    sum += data.instance(j).value(i);
                                }
                            }
                            double mean = sum/data.numInstances();
                            in.setValue(i, mean);
                        }
                    }
                }
            }
            result.add(in);
        }

        return result;
    }

    @Override
    public void buildClassifier(Instances instances) throws Exception {
        Instances data = getDataWithoutMissingValues(instances);
        getCapabilities().testWithFail(data);

        instances = new Instances(data);
        instances.deleteWithMissingClass();

        att_thresholds = new double[instances.numAttributes()];
        att_thresholds = getAttributeThresholds(instances);

        makeTree(instances, instances.numAttributes());
        makeRules();
//        for (double[] rule : rules) {
//            for (double d : rule) {
//                System.out.println(d);
//            }
//            System.out.println();
//        }
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        if (instance.hasMissingValue()) {

        }

        if (attribute == null) {
            return classValue;
        } else {
            if (attribute.isNumeric()) {
                return nodes[((int) getThresholdedValueAttribute(instance, attribute))].classifyInstance(instance);
            } else {
                return nodes[(int) instance.value(attribute)].classifyInstance(instance);
            }
        }
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        if (instance.hasMissingValue()) {

        }

        if (attribute == null) {
            return classDistribution;
        } else {
            if (attribute.isNumeric()) {
                return nodes[((int) getThresholdedValueAttribute(instance, attribute))].distributionForInstance(instance);
            } else {
                return nodes[(int) instance.value(attribute)].distributionForInstance(instance);
            }

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

    private double log2(double value) throws Exception {
        return Math.log(value) / Math.log(2);
    }

    private boolean isEqual(double a, double b) throws Exception {
        return (a - b < EPSILON) && (b - a < EPSILON);
    }

    private double getThresholdedValueAttribute(Instance in, Attribute att) throws Exception {
//        System.out.print(in.value(att) + " ");
//        System.out.println(att.index());
//        if (in.value(att) < att_thresholds[att.index()]) {
//            return 0.0;
//        } else {
//            return 1.0;
//        }
        return 1.0;
    }

    private Instances[] getSplittedData(Instances data, Attribute att) throws Exception {
        Instances[] splits;

        if (att.isNumeric()) {
            splits = new Instances[2];

            for (int i = 0; i < 2; ++i) {
                splits[i] = new Instances(data, data.numInstances());
            }

            Enumeration<Instance> enumInst = data.enumerateInstances();
            while (enumInst.hasMoreElements()) {
                Instance in = enumInst.nextElement();
                splits[((int) getThresholdedValueAttribute(in, att))].add(in);
            }
        } else {
            splits = new Instances[att.numValues()];

            for (int i = 0; i < att.numValues(); ++i) {
                splits[i] = new Instances(data, data.numInstances());
            }

            Enumeration<Instance> enumInst = data.enumerateInstances();
            while (enumInst.hasMoreElements()) {
                Instance in = enumInst.nextElement();
                splits[((int) in.value(att))].add(in);
            }
        }

        for(int i = 0; i < splits.length; ++i) {
            splits[i].compactify();
        }

        return splits;
    }


    private double getEntropy(Instances data) throws Exception {
        double entropy = 0.0;
        double[] countClassValues = new double[data.numClasses()];

        Enumeration<Instance> enumInst = data.enumerateInstances();
        while (enumInst.hasMoreElements()) {
            Instance in = enumInst.nextElement();
            countClassValues[((int) in.classValue())]++;
        }

        for (int i = 0; i < data.numClasses(); ++i) {
            if (countClassValues[i] > 0) {
                double probability = countClassValues[i]/data.numInstances();
                entropy -= probability * log2(probability);
            }
        }

        return entropy;
    }

    private double getInformationGain(Instances data, Attribute att) throws Exception {
        double information_gain = getEntropy(data);
        Instances[] splits = getSplittedData(data, att);

        for (Instances split: splits) {
            if (split.numInstances() > 0) {
                information_gain -= ((double) split.numInstances() / (double) data.numInstances()) * getEntropy(split);
            }
        }

        return information_gain;
    }

    private int getMaxIndex(double[] array) throws Exception {
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

    private void normalize(double[] ds) {
        double sum = 0;
        for (double d : ds) {
            sum += d;
        }

        for (int i = 0; i < ds.length; ++i) {
            ds[i] /= sum;
        }
    }

    private double[] getClassDistribution(Instances data) throws Exception{
        double [] result = new double[data.numClasses()];

        Enumeration<Instance> enumInst = data.enumerateInstances();
        while (enumInst.hasMoreElements()) {
            Instance in = enumInst.nextElement();
            result[((int) in.classValue())]++;
        }

        return result;
    }

    private boolean areAllExamplesHaveSameClassValue(double[] ds) throws Exception {
        int zero = 0;

        for (double d : ds) {
            if (d == 0) {
                ++zero;
            }
        }

        return (zero == ds.length-1);
    }

    private void makeTree(Instances data, int countAtt) throws Exception {
        attribute = null;
        classDistribution = getClassDistribution(data);

        if ((countAtt == 0) || (areAllExamplesHaveSameClassValue(classDistribution))) {
            normalize(classDistribution);
            classValue = getMaxIndex(classDistribution);
            classAttribute = data.classAttribute();
        } else {
            information_gains = new double[data.numAttributes()];
            Enumeration<Attribute> enumAttribute = data.enumerateAttributes();
            while (enumAttribute.hasMoreElements()) {
                Attribute att = enumAttribute.nextElement();
                information_gains[att.index()] = getInformationGain(data, att);
            }
            attribute = data.attribute(getMaxIndex(information_gains));
            System.out.println(getMaxIndex(information_gains));

//            gain_ratios = new double[data.numAttributes()];
//            Enumeration<Attribute> enumAttribute = data.enumerateAttributes();
//            while (enumAttribute.hasMoreElements()) {
//                Attribute att = enumAttribute.nextElement();
//                gain_ratios[att.index()] = getGainRatio(data, att);
//            }
//            attribute = data.attribute(getMaxIndex(gain_ratios));
//            System.out.println(getMaxIndex(gain_ratios));

            Instances[] splits = getSplittedData(data, attribute);
            if (attribute.isNumeric()) {
                nodes = new MyC45[2];
            } else {
                nodes = new MyC45[attribute.numValues()];
            }

            for (int i = 0; i < splits.length; ++i) {
                nodes[i] = new MyC45();
                if (splits[i].isEmpty()) {
                    nodes[i].attribute = null;
                    nodes[i].classDistribution = classDistribution;
                    nodes[i].normalize(nodes[i].classDistribution);
                    nodes[i].classValue = getMaxIndex(nodes[i].classDistribution);
                    nodes[i].classAttribute = data.classAttribute();
                } else {
                    nodes[i].makeTree(splits[i], countAtt-1);
                }
            }
        }
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

    private void sortInstance(Instance[] inst, int index) throws Exception{
        for (int i = 0; i < inst.length; ++i) {
            int k = i;
            for (int j = i+1; j < inst.length; ++j) {
                if (inst[k].value(index) > inst[j].value(index)) {
                    k = j;
                }
            }

            Instance temp = inst[k];
            inst[k] = inst[i];
            inst[i] = temp;
        }
    }

    private double[] getCandidateThresholds(Instance[] data, int index) throws Exception{
        double[] candidate_thresholds = new double[10];
        int counter = -1;
        double mean;
        for (int i = 0; i < data.length-1; ++i) {
            if (counter >= 9) {
                break;
            }
            if (data[i].classValue() != data[i+1].classValue()) {
                mean = (data[i].value(index) + data[i].value(index))/2;
                ++counter;
                candidate_thresholds[counter] = mean;
            }
        }

        return candidate_thresholds;
    }

    private double getMaxInformationGain(double[] candidate, Instances data, int index) throws Exception {
        double result = 0.0;
        double information_gain;
        double selected_information_gain = 0.0;

        for (int i = 0; i < candidate.length; ++i) {
            att_thresholds[index] = candidate[i];
            information_gain = getInformationGain(data, data.attribute(index));
            if (i == 0){
                selected_information_gain = information_gain;
                result = candidate[i];
            }
            if (information_gain > selected_information_gain){
                selected_information_gain = information_gain;
                result = candidate[i];
            }
        }

        return result;
    }

    private double[] getAttributeThresholds(Instances data) throws Exception {
        double[] result = new double[data.numAttributes()];

        for (int i = 0; i < data.numAttributes(); ++i) {
            if (data.attribute(i).isNumeric()) {
                Instance[] insts = new Instance[data.numInstances()];
                for (int j = 0; j < data.numInstances(); ++j ){
                    insts[j] = data.instance(j);
                }
                sortInstance(insts, i);
                double[] candidate_thresholds = getCandidateThresholds(insts, i);
                result[i] = getMaxInformationGain(candidate_thresholds, data, i);
            }
        }

        return result;
    }
}
