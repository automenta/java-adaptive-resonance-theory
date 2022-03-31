package com.github.chen0040.art;

import java.util.ArrayList;
import java.util.List;


/**
 * Created by xschen on 23/8/15.
 */
public class ARTMAP<Y> extends FuzzyART {
    private final List<Y> labels;
    private double epsilon = 0.00001;
    public void setEpsilon(double epsilon) {
        this.epsilon = epsilon;
    }

    public ARTMAP(int inputCount) {
        super(inputCount, 0);
        labels = new ArrayList<>(inputCount);
    }

    public ARTMAP() {
        this(0);
    }

    public Y put(double[] x, Y y, boolean can_create_new_node) {
        int C = nodeCount();


        if (y != null && !labels.contains(y)) {
            addNode(x, y);
        } else {
            if (y!=null && can_create_new_node) {
                boolean new_node = true;
                chooseActivation(x, C);

                for (int j = 0; j < C; j++) {
                    int J = templateActive();
                    if (J == -1) break;

                    Y labelJ = labels.get(J);
                    if (!labelJ.equals(y)) {
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

                if (new_node)
                    addNode(x, y);

            } else {
                Y winner = y;
                double max_match_value = 0;
                int J = -1;
                for (int j = 0; j < C; ++j) {
                    double match_value = match(x, j);
                    if (max_match_value < match_value) {
                        max_match_value = match_value;
                        J = j;
                    }
                }
                return labels.get(J);
            }
        }

        return y;
    }

    private void addNode(double[] x, Y y) {
        addNode(x);
        labels.add(y);
    }

    private void chooseActivation(double[] x, int C) {
        for (int j = 0; j < C; j++)
            activation.set(j, choose(x, j));
    }
}