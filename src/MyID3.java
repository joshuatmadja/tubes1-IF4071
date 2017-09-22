import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.classifiers.AbstractClassifier;
import weka.core.Capabilities.Capability;

import java.util.Enumeration;

public class MyID3 extends AbstractClassifier {

    /**
     * EPSILON untuk menunjukkan dua double bernilai sama
     */
    private static final double EPSILON = 1e-6;

    @Override
    public void buildClassifier(Instances instances) throws Exception {
        getCapabilities().testWithFail(instances);

        instances = new Instances(instances);
        instances.deleteWithMissingClass();

        makeTree(instances);
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

    /**
     * Method yang berfungsi untuk membangkitkan pohon
     *
     * @param instances examples dari data
     * @throws Exception
     */
    public void makeTree(Instances instances) throws Exception {

    }

    /**
     * Method yang akan mengembalikan nilai log2(x)
     *
     * @param x nilai input
     * @return log2(x)
     */
    public double log2(double x) {
        return Math.log(x) / Math.log(2);
    }

    public boolean equals(double a, double b) {
        return (a - b < EPSILON) && (b - a < EPSILON);
    }

    private double hitungEntropi(Instances data) throws Exception {
        double entropi = 0;
        double[] totalAtributKelas = new double[data.numClasses()];
        Enumeration cacahInst = data.enumerateInstances();
        while (cacahInst.hasMoreElements()) {
            Instance in = (Instance) cacahInst.nextElement();
            totalAtributKelas[(int) in.classValue()]++;
        }

        for (int i = 0; i < data.numClasses(); i++) {
            if (totalAtributKelas[i] > 0) {
                double peluang = totalAtributKelas[i] / data.numInstances();
                entropi -= peluang * log2(peluang);
            }
        }

        return entropi;
    }

}
