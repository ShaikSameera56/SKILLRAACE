import java.util.*;

public class CustomNaiveBayes {

    private Map<String, Map<String, Integer>> classFeatureCounts;
    private Map<String, Integer> classDocumentCounts;
    private int totalDocuments;
    private Set<String> vocabulary;

    public CustomNaiveBayes(List<CustomDocument> documents) {
        classFeatureCounts = new HashMap<>();
        classDocumentCounts = new HashMap<>();
        totalDocuments = documents.size();
        vocabulary = new HashSet<>();

        // Train the classifier with documents
        for (CustomDocument customDoc : documents) {
            String className = customDoc.getCategory();
            List<String> features = customDoc.getWords();

            // Update class document count
            classDocumentCounts.put(className, classDocumentCounts.getOrDefault(className, 0) + 1);

            // Update class feature count
            if (!classFeatureCounts.containsKey(className)) {
                classFeatureCounts.put(className, new HashMap<>());
            }
            Map<String, Integer> featureCounts = classFeatureCounts.get(className);
            for (String feature : features) {
                featureCounts.put(feature, featureCounts.getOrDefault(feature, 0) + 1);
                vocabulary.add(feature);
            }
        }
    }

    public String classify(CustomDocument customDoc) {
        List<String> features = customDoc.getWords();
        double maxProbability = Double.NEGATIVE_INFINITY;
        String predictedCategory = null;

        // Calculate probability for each class
        for (String className : classDocumentCounts.keySet()) {
            double classProbability = Math.log(classDocumentCounts.get(className) / (double) totalDocuments);
            for (String feature : features) {
                int featureCount = classFeatureCounts.get(className).getOrDefault(feature, 0);
                double featureProbability = Math.log((featureCount + 1.0) / // Add-one smoothing
                                                    (classFeatureCounts.get(className).size() + vocabulary.size()));
                classProbability += featureProbability;
            }

            if (classProbability > maxProbability) {
                maxProbability = classProbability;
                predictedCategory = className;
            }
        }

        return predictedCategory;
    }

    public double calculateAccuracy(List<CustomDocument> documents) {
        int correct = 0;
        for (CustomDocument customDoc : documents) {
            String predictedCategory = classify(customDoc);
            if (predictedCategory.equals(customDoc.getCategory())) {
                correct++;
            }
        }
        return (double) correct / documents.size();
    }

    public double calculatePrecision(String category, List<CustomDocument> documents) {
        int truePositives = 0;
        int predictedPositives = 0;
        for (CustomDocument customDoc : documents) {
            if (classify(customDoc).equals(category)) {
                predictedPositives++;
                if (customDoc.getCategory().equals(category)) {
                    truePositives++;
                }
            }
        }
        return predictedPositives > 0 ? (double) truePositives / predictedPositives : 0.0;
    }

    public double calculateRecall(String category, List<CustomDocument> documents) {
        int truePositives = 0;
        int actualPositives = 0;
        for (CustomDocument customDoc : documents) {
            if (customDoc.getCategory().equals(category)) {
                actualPositives++;
                if (classify(customDoc).equals(category)) {
                    truePositives++;
                }
            }
        }
        return actualPositives > 0 ? (double) truePositives / actualPositives : 0.0;
    }

    // Implement CustomDocument class to represent a document with features and class label
    public static class CustomDocument {
        private String category;
        private List<String> words;

        public CustomDocument(String category, List<String> words) {
            this.category = category;
            this.words = words;
        }

        public String getCategory() {
            return category;
        }

        public List<String> getWords() {
            return words;
        }
    }

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);

        // Input training documents
        System.out.println("Enter the number of training documents:");
        int numDocuments = Integer.parseInt(scanner.nextLine());
        List<CustomDocument> documents = new ArrayList<>();

        for (int i = 0; i < numDocuments; i++) {
            System.out.println("Enter category for document " + (i + 1) + ":");
            String category = scanner.nextLine();
            System.out.println("Enter words (space-separated) for document " + (i + 1) + ":");
            List<String> words = Arrays.asList(scanner.nextLine().split(" "));
            documents.add(new CustomDocument(category, words));
        }

        // Initialize classifier with training documents
        CustomNaiveBayes classifier = new CustomNaiveBayes(documents);

        // Input test document
        System.out.println("Enter words (space-separated) for the test document:");
        List<String> testWords = Arrays.asList(scanner.nextLine().split(" "));
        CustomDocument testDoc = new CustomDocument("Unknown", testWords);

        // Classify the test document
        String predictedCategory = classifier.classify(testDoc);
        System.out.println("Predicted category for the test document: " + predictedCategory);

        // Calculate accuracy
        double accuracy = classifier.calculateAccuracy(documents);
        System.out.println("Accuracy: " + accuracy);

        // Calculate precision and recall for a specific category
        System.out.println("Enter the category to calculate precision and recall:");
        String category = scanner.nextLine();
        double precision = classifier.calculatePrecision(category, documents);
        double recall = classifier.calculateRecall(category, documents);
        System.out.println("Precision for category " + category + ": " + precision);
        System.out.println("Recall for category " + category + ": " + recall);

        scanner.close();
    }
}
