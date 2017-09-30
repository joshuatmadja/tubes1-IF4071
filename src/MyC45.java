import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.classifiers.AbstractClassifier;
import weka.core.Capabilities.Capability;

import java.util.Enumeration;

public class MyC45 extends AbstractClassifier {
    private static final double EPSILON = 1e-6;
    private MyC45[] nodes;
    private double[] information_gains;
    private double[] classDistribution;
    private double classValue;
    private Attribute attribute;
    private Attribute classAttribute;

    @Override
    public void buildClassifier(Instances instances) throws Exception {
        getCapabilities().testWithFail(instances);

        instances = new Instances(instances);
        instances.deleteWithMissingClass();

//        makeTreeOld(instances);
        makeTree(instances, instances.numAttributes());
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        if (instance.hasMissingValue()) {

        }

        if (attribute == null) {
            return classValue;
        } else {
            return nodes[(int) instance.value(attribute)].classifyInstance(instance);
        }
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        if (instance.hasMissingValue()) {

        }

        if (attribute == null) {
            return classDistribution;
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

    private Instances[] getSplittedData(Instances data, Attribute att) throws Exception {
        Instances[] splits = new Instances[att.numValues()];

        for (int i = 0; i < att.numValues(); ++i) {
            splits[i] = new Instances(data, data.numInstances());
        }

        Enumeration<Instance> enumInst = data.enumerateInstances();
        while (enumInst.hasMoreElements()) {
            Instance in = enumInst.nextElement();
            splits[((int) in.value(att))].add(in);
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

            Instances[] splits = getSplittedData(data, attribute);
            nodes = new MyC45[attribute.numValues()];
            for (int i = 0; i < attribute.numValues(); ++i) {
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
}
