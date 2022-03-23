package com.github.chen0040.art;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class ART1 {

    /** choice parameter */
    protected double alpha;

    /** base resonance threshold */
    protected double rho;

    /** learning rate */
    protected double beta;

    protected List<double[]> weights;

    /** activation values */
    protected List<Double> activation;

    protected int inputCount;

    public ART1() {
        this(0, 0);
    }

    public ART1(int inputCount, int initialNeuronCount) {
        this.inputCount = inputCount;

        weights = new ArrayList<>(/*?*/);
        activation = new ArrayList<>(/*?*/);

        for (int i = 0; i < initialNeuronCount; ++i)
            addNode();
    }

    public void setAlpha(double alpha) {
        this.alpha = alpha;
    }

    public void setRho(double rho) {
        this.rho = rho;
    }

    public void setBeta(double beta) {
        this.beta = beta;
    }

    public void addNode() {
        double[] neuron = new double[inputCount];
        Arrays.fill(neuron, 1);
        addNodeDirect(neuron);
    }
    public void addNode(double[] x) {
        addNodeDirect(x.clone());
    }

    public final void addNodeDirect(double[] x) {
        weights.add(x);
        activation.add(0.0);
    }

    protected double choice(double[] x, int j) {
        double[] W_j = weights.get(j);
        double sum = 0, sum2 = 0;
        for (int i = 0; i < x.length; ++i) {
            double Wji = W_j[i];
            sum += Math.abs(x[i] * Wji); // norm1
            sum2 += Math.abs(Wji); //  norm1
        }
        return sum / (alpha + sum2);
    }

    /** template with max activation */
    protected int templateActive() {
        int C = nodeCount();
        double vMax = Double.NEGATIVE_INFINITY;
        int t = -1;
        for (int i = 0; i < C; ++i) {
            double v = activation.get(i);
            if (v > vMax) {
                vMax = v;
                t = i;
            }
        }
        return t;
    }

    public int nodeCount() {
        return weights.size();
    }

    protected double match(double[] x, int j) {
        double sum = 0, sum2 = 0;
        double[] W_j = weights.get(j);
        for (int i = 0; i < x.length; ++i) {
            double Xi = x[i];
            sum += Math.abs(Xi * W_j[i]); // norm1
            sum2 += Math.abs(Xi); //  norm1
        }
        return sum / sum2;
    }

    protected void updateNode(double[] x, int j) {
        double[] W_j = weights.get(j);
        for (int i = 0; i < x.length; ++i) {
            double Wji = W_j[i];
            W_j[i] = (1 - beta) * Wji + beta * Wji * x[i];
        }
    }

    public int simulate(double[] x, boolean can_create_new_node) {
        boolean new_node = can_create_new_node;
        int C = nodeCount();

        int winner = -1;

        if (can_create_new_node) {
            for (int i = 0; i < C; ++i) {
                activation.set(i,
                    choice(x, i));
            }

            for (int i = 0; i < C; ++i) {
                int J = templateActive();
                if (J == -1) break;

                double match_value = match(x, J);
                if (match_value > rho) {
                    updateNode(x, J);
                    winner = J;
                    new_node = false;
                    break;
                } else {
                    activation.set(J, 0.0);
                }
            }

            if (new_node) {
                addNode(x);
                winner = nodeCount() - 1;
            }
        } else {
            double max_match_value = Double.NEGATIVE_INFINITY;
            int J = -1;
            for (int j = 0; j < C; ++j) {
                double match_value = match(x, j);
                if (max_match_value < match_value) {
                    max_match_value = match_value;
                    J = j;
                }
            }
            winner = J;
        }

        return winner;
    }
}