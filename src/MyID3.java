import weka.classifiers.Classifier;
import weka.core.Attribute;
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
    private MyID3[] akar;
    private Attribute atribut;
    private double kelas;
    private double[] persebaranKelas;
    private Attribute atributKelas;

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
    public void makeTree(Instances data) throws Exception {
        //periksa jika example kosong
        if (data.numInstances() == 0) {
            atribut = null;
            kelas = Double.NaN;
            persebaranKelas = new double[data.numClasses()];
            return;
        }

        //cek info gain tertinggi kemudian dipilih sebagai akar
        double[] gains = new double[data.numAttributes()];
        Enumeration cacahAtribut = data.enumerateAttributes();
        while (cacahAtribut.hasMoreElements()) {
            Attribute a = (Attribute) cacahAtribut.nextElement();
            gains[a.index()] = hitungInfoGain(data, a);
        }
        atribut = data.attribute(indeksMaksimum(gains));

        if (equals(gains[atribut.index()], 0)) {
            atribut = null;
            persebaranKelas = new double[data.numClasses()];
            Enumeration cacahInstance = data.enumerateInstances();
            while (cacahInstance.hasMoreElements()) {
                Instance in = (Instance) cacahInstance.nextElement();
                persebaranKelas[(int) in.classValue()]++;
            }
            normalisasi(persebaranKelas);
            kelas = indeksMaksimum(persebaranKelas);
            atributKelas = data.classAttribute();
        } else {
            Instances[] split = pisahData(data, atribut);
            akar = new MyID3[atribut.numValues()];
            for (int i = 0; i < atribut.numValues(); i++) {
                akar[i] = new MyID3();
                akar[i].makeTree(split[i]);
            }
        }
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

    private double hitungInfoGain(Instances data, Attribute a) throws Exception {
        double infoGain = hitungEntropi(data);
        Instances[] split = pisahData(data, a);

        for (int i = 0; i < a.numValues(); i++) {
            if (split[i].numInstances() > 0) {
                infoGain -= ((double) split[i].numInstances() / (double) data.numInstances()) * hitungEntropi(split[i]);
            }
        }
        return infoGain;
    }

    private int indeksMaksimum(double[] array) {
        double max = array[0];
        int idx = 0;

        for (int i = 1; i < array.length; i++) {
            if (array[i] > max) {
                idx = i;
                max = array[i];
            }
        }

        return idx;
    }

    private Instances[] pisahData(Instances data, Attribute a) {
        Instances[] split = new Instances[a.numValues()];
        for (int i = 0; i < a.numValues(); i++) {
            split[i] = new Instances(data, data.numInstances());
        }

        Enumeration cacahInstance = data.enumerateAttributes();
        while (cacahInstance.hasMoreElements()) {
            Instance in = (Instance) cacahInstance.nextElement();
            split[(int) in.value(a)].add(in);
        }

        for (int i = 0; i < split.length; i++) {
            split[i].compactify();
        }

        return split;
    }

    private void normalisasi(double[] ds) {
        double sum = 0;
        for (double d : ds) {
            sum += d;
        }

        for (int i = 0; i < ds.length; i++) {
            ds[i] /= sum;
        }
    }
}
