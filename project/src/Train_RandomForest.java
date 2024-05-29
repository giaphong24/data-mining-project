import java.util.HashMap;
import java.util.Random;

import weka.core.Instances;
import weka.classifiers.Evaluation;
import weka.core.Utils;
import weka.core.WekaPackageManager;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.SerializationHelper;
import weka.classifiers.trees.RandomForest;

public class Train_RandomForest {
    public static void main(String[] args) throws Exception {
        WekaPackageManager.loadPackages(false);
        ArgumentParser parser = new ArgumentParser(args);
        String DATASETPATH = parser.GetIndex().get(0);

        // 0. Load dataset
        System.out.println("Loading dataset ...");
        DataSource source = new DataSource(DATASETPATH);
        Instances dataset = source.getDataSet();
        // Class index is the attribute named "Category"
        for (int i = 0; i < dataset.numAttributes(); i++) {
            if (dataset.attribute(i).name().equals("Category")) {
                dataset.setClassIndex(i);
                break;
            }
        }
        // Integer num_predictors = dataset.numAttributes() - 1;

        // 1.1 Initialize Random Forest
        RandomForest rf = new RandomForest();

        // 1.2 Setup default parameters for Random Forest that have zero impact on the
        // model performance
        rf.setCalcOutOfBag(true);
        rf.setPrintClassifiers(true);
        rf.setOutputOutOfBagComplexityStatistics(true);
        rf.setStoreOutOfBagPredictions(true);
        rf.setComputeAttributeImportance(true);
        rf.setNumExecutionSlots(0);
        rf.setNumDecimalPlaces(3);

        // 1.3 Now get the user-defined parameters
        HashMap<String, String> named_arguments = parser.GetNamedArguments();

        String num_features = named_arguments.get("-K") == null ? "0" : named_arguments.get("-K");
        rf.setNumFeatures(Integer.parseInt(num_features));

        // Max depth of the tree, default: 0 = no maximum depth. But better to use 3
        String max_depth = named_arguments.get("-depth") == null ? "3" : named_arguments.get("-depth");
        rf.setMaxDepth(Integer.parseInt(max_depth));

        // Number of trees, default: 100
        String num_trees = named_arguments.get("-I") == null ? "300" : named_arguments.get("-I");
        rf.setNumIterations(Integer.parseInt(num_trees));

        // Random Seed, default: 1
        String seed = named_arguments.get("-S") == null ? "1" : named_arguments.get("-S");
        rf.setSeed(Integer.parseInt(seed));

        // Preview all the options
        System.out.println("Random Forest options: " + Utils.joinOptions(rf.getOptions()));

        // 2.0 Now train the model
        System.out.println("Training the model ...");
        // Default: 10-fold cross-validation
        Integer cv_seed = Integer
                .parseInt(named_arguments.get("-cv_seed") == null ? "0" : named_arguments.get("-cv_seed"));
        Integer cv_folds = Integer
                .parseInt(named_arguments.get("-cv_folds") == null ? "10" : named_arguments.get("-cv_folds"));
        // rf.buildClassifier(dataset); // I don't know why we need to build the
        // classifier first before cross-validation

        Evaluation eval = new Evaluation(dataset);
        eval.crossValidateModel(rf, dataset, cv_folds, new Random(cv_seed));

        // 3.0 Print the results
        System.out.println("===== Random Forest =====");
        System.out.println("Classifier: " + rf.getClass().getName());
        System.out.println("Dataset: " + dataset.relationName());
        System.out.println("Number of instances: " + dataset.numInstances());
        System.out.println("Number of attributes: " + dataset.numAttributes());
        System.out.println("Number of classes: " + dataset.numClasses());
        System.out.println("Number of CV Folds: " + cv_folds);
        System.out.println("CV Seed: " + cv_seed);
        System.out.println("Options: " + Utils.joinOptions(rf.getOptions()));
        System.out.println("Attribute Importance" + rf.computeAttributeImportanceTipText());

        // Performance metrics
        System.out.println(eval.toSummaryString("=== " + cv_folds + "-fold Cross-validation ===", true));
        System.out.println("Average Cost for Misclassification: " + eval.avgCost());
        System.out.println("Total Cost: " + eval.totalCost());

        // Confusion matrix
        System.out.println(eval.toMatrixString("=== Confusion Matrix ==="));
        // Class details
        System.out.println(eval.toClassDetailsString("=== Class Details ==="));

        // 4.0 Save the model
        String model_path = named_arguments.get("--output");
        if (model_path != null) {
            if (!model_path.endsWith(".model")) {
                model_path += ".model";
            }
            System.out.println("Saving the model to " + model_path);
            System.out.println(
                    "Warning: Since no independent testing set is provided, the model is saved without cross-validation.");
            rf.buildClassifier(dataset);
            SerializationHelper.write(model_path, rf);
        }

    }
}
