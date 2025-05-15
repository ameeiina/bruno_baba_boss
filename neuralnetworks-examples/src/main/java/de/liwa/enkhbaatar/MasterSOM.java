package de.liwa.enkhbaatar;

import com.brunomnsilva.neuralnetworks.core.ConsoleProgressBar;
import com.brunomnsilva.neuralnetworks.core.VectorN;
import com.brunomnsilva.neuralnetworks.dataset.*;
import com.brunomnsilva.neuralnetworks.models.som.*;
import com.brunomnsilva.neuralnetworks.models.som.impl.UbiSOM;
import com.brunomnsilva.neuralnetworks.view.GenericWindow;
import com.brunomnsilva.neuralnetworks.view.som.SelfOrganizingMapVisualizationFactory;
import com.brunomnsilva.yacl.core.Clusterable;
import com.brunomnsilva.yacl.hierarchical.Dendogram;
import com.brunomnsilva.yacl.hierarchical.HierarchicalClustering;
import com.brunomnsilva.yacl.hierarchical.HierarchicalClusteringResult;
import com.brunomnsilva.yacl.view.DendogramVisualization;

import javax.swing.*;
import java.awt.*;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

public class MasterSOM {
    /**
     * This class wraps a component plane instance and provides the clusterable behavior needed for the feature
     * clustering procedure, utilizing the YACL clustering library.
     */
    private static class ComponentPlaneWrapper implements Clusterable<ComponentPlaneWrapper> {
        private final ComponentPlane componentPlane;

        public ComponentPlaneWrapper(ComponentPlane componentPlane) {
            this.componentPlane = componentPlane;
        }

        private static double cosineDistance(ComponentPlane cp1, ComponentPlane cp2) {
            VectorN Ci = VectorN.fromArray(cp1.flatten());
            VectorN Cj = VectorN.fromArray(cp2.flatten());
            double dot = Ci.dot(Cj);
            double magCi = Ci.magnitude();
            double magCj = Cj.magnitude();
            double similarity = dot / (magCi * magCj); // -1 to 1
            return 1 - similarity;
        }

        private static double euclideanDistance(ComponentPlane cp1, ComponentPlane cp2) {
            VectorN Ci = VectorN.fromArray(cp1.flatten());
            VectorN Cj = VectorN.fromArray(cp2.flatten());
            return Ci.distance(Cj);
        }

        private static double pearsonDistance(ComponentPlane cp1, ComponentPlane cp2) {
            VectorN Ci = VectorN.fromArray(cp1.flatten());
            VectorN Cj = VectorN.fromArray(cp2.flatten());
            VectorN meanCi = VectorN.rep(Ci.dimensions(), Ci.mean());
            VectorN meanCj = VectorN.rep(Cj.dimensions(), Cj.mean());
            double stdCi = Ci.std();
            double stdCj = Cj.std();
            // adjust flattened vectors
            Ci.subtract(meanCi);
            Ci.divide(stdCi);
            Cj.subtract(meanCj);
            Cj.divide(stdCj);
            double dot = Ci.dot(Cj);
            double similarity = dot / Ci.dimensions(); // -1 to 1
            return 1 - similarity;
        }

        @Override
        public double clusterableDistance(ComponentPlaneWrapper other) {
            // This must implement some kind of metric distance to be used by the hierarchical clustering
            // procedure. Below are three possible implementation examples you can choose from. You can derive your own.
            return pearsonDistance(this.componentPlane, other.componentPlane);
            //return euclideanDistance(this.componentPlane, other.componentPlane);
            //return cosineDistance(this.componentPlane, other.componentPlane);
        }

        @Override
        public String clusterableLabel() {
            return componentPlane.getName();
        }

        @Override
        public double[] clusterablePoint() {
            return componentPlane.flatten();
        }
    }

    private static void featureClustering(SelfOrganizingMap som, String[] inputNames) {
        int dimensionality = som.getDimensionality();
        List<ComponentPlaneWrapper> components = new ArrayList<>();
        // Create the component planes
        for (int d = 0; d < dimensionality; ++d) {
            ComponentPlane cp = ComponentPlane.fromSelfOrganizingMap(som, d, inputNames[d]);
            components.add(new ComponentPlaneWrapper(cp));
        }
        // Hierarchical clustering usage
        HierarchicalClustering<ComponentPlaneWrapper> hclust = new HierarchicalClustering<>("complete");
        HierarchicalClusteringResult<ComponentPlaneWrapper> hclustResult = hclust.cluster(components);
        Dendogram<ComponentPlaneWrapper> dendogram = new Dendogram<>(hclustResult);
        DendogramVisualization<ComponentPlaneWrapper> viz = new DendogramVisualization<>(dendogram, DendogramVisualization.LabelType.LABEL);
        GenericWindow window = GenericWindow.horizontalLayout("Feature Clustering", viz);
        window.setPreferredSize(new Dimension(1000, 800));
        window.pack();
        window.exitOnClose();
        window.setVisible(true);
    }

    public static void main(String[] args) {
        try {
            // Load a dataset and normalize it
            //Dataset dataset = new Dataset("brunos-datasets/wine.data");
            Dataset dataset = new Dataset("run1_2000_50.data");
            dataset.shuffle();
            DatasetNormalization normalization = new MinMaxNormalization(dataset);
            // normalization.normalize(dataset);
            // Create basic SOM with random initialization of prototypes
            int width = 40;
            int height = 20;
            UbiSOM som = new UbiSOM(width, height, dataset.inputDimensionality(), new SimpleRectangularLattice(), new EuclideanDistance(12), 0.1,
                    0.08, 0.6, 0.2, 0.7, 2000);
            int orderEpochs = 100;
            try {
                som.load(Path.of("models"), "som");
            } catch (IOException _) {
            }
            ConsoleProgressBar progress = new ConsoleProgressBar(orderEpochs);
            int epochCount = 1;
            for (int i = 0; i < orderEpochs; ++i) {
                for (DatasetItem datasetItem : dataset) {
                    som.learn(datasetItem.getInput());
                }
                progress.update(epochCount++);
            }
            System.out.println();
            som.save(Path.of("models"), "som");
            som.print(Path.of("models"), "som");
            // Print statistics for model fitting
            SelfOrganizingMapStatistics statistics = SelfOrganizingMapStatistics.compute(som, dataset);
            System.out.println(statistics);
            showVisualizations(som, dataset);
            //showHitVisualisation(som, dataset);
            //showTargetVisualisation(som, dataset);
            //featureClustering(som, dataset.inputVariableNames());
        } catch (IOException | InvalidDatasetFormatException e) {
            e.printStackTrace();
        }
    }

    private static void showHitVisualisation(SelfOrganizingMap som, Dataset dataset) {
        JPanel[] panels = new JPanel[1];
        panels[0] = SelfOrganizingMapVisualizationFactory.createHitMap(som, dataset);
        GenericWindow window = GenericWindow.gridLayout("Hit-Matrix", 1, 1, panels);
        window.exitOnClose();
        window.setVisible(true);
    }

    private static void showTargetVisualisation(SelfOrganizingMap som, Dataset dataset) {
        JPanel[] panels = new JPanel[1];
        // U-Matrix
        panels[0] = SelfOrganizingMapVisualizationFactory.createTargetOutputProjection(som, dataset);
        GenericWindow window = GenericWindow.gridLayout("U-Matrix", 1, 1, panels);
        window.exitOnClose();
        window.setVisible(true);
    }

    private static void showVisualization(SelfOrganizingMap som, Dataset dataset) {
        JPanel[] panels = new JPanel[1];
        // Create the U-Matrix
        panels[0] = SelfOrganizingMapVisualizationFactory.createUMatrix(som);
        GenericWindow window = GenericWindow.gridLayout("U-Matrix", 1, 1, panels);
        window.exitOnClose();
        window.setVisible(true);
    }

    private static void showVisualizations(SelfOrganizingMap som, Dataset dataset) {
        int dimensionality = som.getDimensionality();
        // Create a set of panels for U-Matrix + all component planes
        JPanel[] panels = new JPanel[dimensionality + 1];
        // Create the U-Matrix
        panels[0] = SelfOrganizingMapVisualizationFactory.createUMatrix(som);
        // Create the component planes
        int d;
        for (d = 0; d < dimensionality; ++d) {
            panels[d + 1] = SelfOrganizingMapVisualizationFactory.createComponentPlane(som, d, dataset.inputVariableNames()[d]);
        }
        GenericWindow window = GenericWindow.gridLayout("U-Matrix and Component Planes", 5, 4, panels);
        window.exitOnClose();
        window.setVisible(true);
    }
}
