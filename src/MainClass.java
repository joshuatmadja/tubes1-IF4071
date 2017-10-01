import java.util.Random;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;
import weka.classifiers.trees.J48;

public class MainClass {

    private static Instances train;
    private static Instances test;
    private static J48 j48;

    public static Instances readFile(String filename) throws Exception {
        DataSource source = new DataSource(filename);
        Instances data = source.getDataSet();
        // setting class attribute if the data format does not provide this information
        // For example, the XRFF format saves the class attribute information as well
        if (data.classIndex() == -1)
            data.setClassIndex(data.numAttributes() - 1);
        return data;
    }

    public static Instances filterResample(Instances data) throws Exception {
        Resample filter = new Resample();
        filter.setBiasToUniformClass(1.0);

        filter.setInputFormat(data);
        filter.setNoReplacement(false);
        filter.setSampleSizePercent(100);
        Instances filteredInstances = Filter.useFilter(data, filter);

        return filteredInstances;
    }


    public static void crossValidation(Classifier cls, int folds) throws Exception{
        Evaluation eval = new Evaluation(train);
        eval.crossValidateModel(cls, train, folds, new Random(1));
        printEval(eval);
    }

    public static void percentageSplit (Classifier cls) throws Exception{
        //random dataset
        train.randomize(new Random(0));
        // 80% split
        int trainSize = (int) Math.round(train.numInstances() * 66 / 100);
        int testSize = train.numInstances() - trainSize;
        Instances train_split = new Instances(train, 0, trainSize);
        test = new Instances(train, trainSize, testSize);

        cls.buildClassifier(train_split);

        Evaluation eval = new Evaluation(test);
        eval.evaluateModel(cls, test);
        printEval(eval);

    }

    public static void trainingTest (Classifier cls) throws Exception{
        cls.buildClassifier(train);
        Evaluation eval = new Evaluation(train);
        eval.evaluateModel(cls, train);
        printEval(eval);

    }

    public static void printEval(Evaluation E) throws Exception {
        System.out.println(E.toSummaryString("\nResults\n======\n", false));
        System.out.println(E.toClassDetailsString());
        System.out.println(E.toMatrixString());
    }

    public static void printAccuracy(Evaluation E) throws Exception {
        System.out.println(E.correct());
    }

    public static void preprocessFile(String fileName) throws Exception {
        Instances data = readFile(fileName);
        Instances filteredData = filterResample(data);
        train = filteredData;
    }

    public static void saveModel(Classifier cls, String filename) throws Exception{
        weka.core.SerializationHelper.write("../" + filename, cls);
    }

    public static Classifier loadModel(String filename) throws Exception{
        return (Classifier) weka.core.SerializationHelper.read(filename);

    }

    public static void main(String[] args) throws Exception {

        MyID3 ID3 = new MyID3();
        MyC45 C45 = new MyC45();
        j48 = new J48();
//        Classifier cls = j48;
//        Classifier cls = ID3;
        Classifier cls = C45;
        preprocessFile("weather.nominal.arff");
//        crossValidation(cls, 10);
//        percentageSplit(cls);
        trainingTest(cls);
    }
}
