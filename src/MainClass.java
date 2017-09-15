import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;

public class MainClass {

    Instances train;
    Instances test;

    public Instances readFile(String filename) throws Exception {
        DataSource source = new DataSource(filename);
        Instances data = source.getDataSet();
        // setting class attribute if the data format does not provide this information
        // For example, the XRFF format saves the class attribute information as well
        if (data.classIndex() == -1)
            data.setClassIndex(data.numAttributes() - 1);
        return data;
    }

    public Instances filterResample(Instances data) throws Exception {
        Resample filter = new Resample();
        filter.setBiasToUniformClass(1.0);

        filter.setInputFormat(data);
        filter.setNoReplacement(false);
        filter.setSampleSizePercent(100);
        Instances filteredInstances = Filter.useFilter(data, filter);

        return filteredInstances;
    }


    public void crossValidateEval(Classifier cls, int folds) {
        // Evaluation eval = new Evaluation();
        // eval.crossValidateModel(cls, );
    }

    public void preprocessFile(String fileName) throws Exception {
        Instances data = readFile(fileName);
        Instances filteredData = filterResample(data);
    }
}
