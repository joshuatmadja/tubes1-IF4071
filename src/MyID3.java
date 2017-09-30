import org.w3c.dom.Attr;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.classifiers.AbstractClassifier;
import weka.core.Capabilities.Capability;

import java.lang.reflect.Array;
import java.util.*;

public class MyID3 extends AbstractClassifier {

    /**
     * EPSILON untuk menunjukkan dua double bernilai sama
     */
    private static final double EPSILON = 1e-6;
    private MyID3[] akar;
    private Attribute atribut;
    private double kelas;
    private double[] persebaranKelas;
    private Attribute atributKelas;

    private HashMap<String, MyID3> trees;

    @Override
    public void buildClassifier(Instances instances) throws Exception {
        getCapabilities().testWithFail(instances);

        instances = new Instances(instances);
        instances.deleteWithMissingClass();

//        makeTree(instances);
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        return 0;
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        return new double[0];
    }

    @Override
    public Capabilities getCapabilities() {
        Capabilities initialCapabilities = super.getCapabilities();
        initialCapabilities.disableAll();

        //untuk atribut
        initialCapabilities.enable(Capability.NOMINAL_ATTRIBUTES);

        //untuk kelas
        initialCapabilities.enable(Capability.NOMINAL_CLASS);
        initialCapabilities.enable(Capability.MISSING_CLASS_VALUES);

        initialCapabilities.setMinimumNumberInstances(0);

        return initialCapabilities;
    }

    public void makeTree(Instances data) throws Exception {

    }

    public double log2(double x) {
        return Math.log(x) / Math.log(2);
    }

    private double x(Instances data) throws Exception {
        double entropy = 0.0;
        HashMap<String, Integer> totalClassAttributes = new HashMap<>();

        for (int i = 0; i < data.numClasses(); ++i) {
//            totalClassAttributes.put(data.attribute("2" , 0)
        }

        return entropy;
    }

    private double calculateEntropy(Instances data) throws Exception {
        double entropy = 0.0;
        double[] totalClassAttributes = new double[data.numClasses()];

        for (int i = 0; i < totalClassAttributes.length; ++i) {
            totalClassAttributes[i] = 0.0;
        }

        Enumeration enumInstance = data.enumerateInstances();
        while (enumInstance.hasMoreElements()) {
            Instance in = (Instance) enumInstance.nextElement();
            totalClassAttributes[(int) in.classValue()]++;
        }

        for (int i = 0; i < data.numClasses(); ++i) {
            if (totalClassAttributes[i] > 0) {
                double prob = totalClassAttributes[i] / data.numInstances();
                entropy -= prob * log2(prob);
            }
        }

        return entropy;
    }

    private double calculateInfoGain(Instances data, Attribute att) throws Exception {
        double info_gain = calculateEntropy(data);
        Instances[] split = seperateData(data, att);

        for (int i = 0; i < split.length; ++i) {
            if (split[i].numInstances() > 0) {
                info_gain -= ((double) split[i].numInstances() / (double) data.numInstances()) * calculateEntropy(split[i]);
            }
        }

        return info_gain;
    }

    private Instances[] seperateData(Instances data, Attribute att) throws Exception {
        Instances[] split = new Instances[att.numValues()];

        Enumeration enumAttribute = data.enumerateAttributes();
        while (enumAttribute.hasMoreElements()) {
            Attribute att_1 = (Attribute) enumAttribute.nextElement();
//            split.get((int) att_1.value())
        }

        return split;
    }

}
