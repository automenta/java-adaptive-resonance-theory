package com.github.chen0040.art;

import java.util.ArrayList;
import java.util.List;


/**
 * Created by xschen on 23/8/15.
 */
public class ARTMAP extends FuzzyART {
    private final List<String> labels;
    private double epsilon = 0.00001;
    public void setEpsilon(double epsilon) {
        this.epsilon = epsilon;
    }

    public ARTMAP(int inputCount) {
        super(inputCount, 0);
        labels = new ArrayList<>();
    }

    public ARTMAP() {
        super();
        labels = new ArrayList<>();
    }

    public String simulate(double[] x, String label, boolean can_create_new_node) {
        boolean new_node = can_create_new_node;
        int C = nodeCount();

        String winner = label;

        if (label != null && !labels.contains(label)) {
            addNode(x);
            labels.add(label);
        } else {
            if (label == null) {
                can_create_new_node = false;
            }

            if (can_create_new_node) {
                for (int j = 0; j < C; ++j) {
                    activation.set(j,
                            choice(x, j));
                }

                for (int j = 0; j < C; ++j) {
                    int J = templateActive();
                    if (J == -1) break;

                    String labelJ = labels.get(J);
                    if (!labelJ.equals(label)) {
                        rho = match(x, J) + epsilon;
                    }

                    double match_value = match(x, J);
                    if (match_value > rho) {
                        updateNode(x, J);
                        new_node = false;
                        break;
                    } else {
                        activation.set(J, 0.0);
                    }
                }

                if (new_node) {
                    addNode(x);
                    labels.add(label);
                }
            } else {
                double max_match_value = 0;
                int J = -1;
                for (int j = 0; j < C; ++j) {
                    double match_value = match(x, j);
                    if (max_match_value < match_value) {
                        max_match_value = match_value;
                        J = j;
                    }
                }
                winner = labels.get(J);
            }
        }

        return winner;
    }
}