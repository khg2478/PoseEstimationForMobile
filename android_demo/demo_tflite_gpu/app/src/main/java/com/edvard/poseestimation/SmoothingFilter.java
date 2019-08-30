package com.edvard.poseestimation;
import org.opencv.core.KeyPoint;

import java.util.ArrayList;
import java.util.Arrays;
import java.lang.Math;

public class SmoothingFilter {

    private int num_frames_for_filter;
    private int num_lowest_removed;
    private ArrayList<ArrayList<Keypoint>> arraylist_all_frames_keypoints;

    SmoothingFilter(int num_frames_for_filter, int num_lowest_removed) {
        this.num_frames_for_filter = num_frames_for_filter;
        this.num_lowest_removed = num_lowest_removed;
        this.arraylist_all_frames_keypoints = new ArrayList<ArrayList<Keypoint>>();
    }

    public float[][] getFilteredKeypoints(float[][][][] heatMapArray) {
        ArrayList<Keypoint> arraylist_keypoints = new ArrayList<Keypoint>();
        int height = heatMapArray[0].length;
        int width = heatMapArray[0][0].length;
        int num_keypoints = heatMapArray[0][0][0].length;
        //get keypoints
        for (int i = 0 ; i < num_keypoints ; i++) {
            float max = 0;
            int maxX = 0;
            int maxY = 0;
            for (int h = 0 ; h < height ; h++) {
                for (int w = 0 ; w < width ; w++) {
                    if (max < heatMapArray[0][h][w][i]) {
                        max = heatMapArray[0][h][w][i];
                        maxX = w;
                        maxY = h;
                    }
                }
            }
            arraylist_keypoints.add(new Keypoint(maxX, maxY, max));
        }
        this.arraylist_all_frames_keypoints.add(arraylist_keypoints);
        if (this.arraylist_all_frames_keypoints.size() <= this.num_frames_for_filter) {
            return get_list_from_arraylist(arraylist_keypoints);
        } else {
            this.arraylist_all_frames_keypoints.remove(0);
            return applyFilter();
        }
    }

    public float[][] get_list_from_arraylist(ArrayList<Keypoint> arraylist_keypoints) {
        float[][] list_keypoints = new float[arraylist_keypoints.size()][3];
        for (int i = 0 ; i < arraylist_keypoints.size() ; i++) {
            list_keypoints[i][0] = arraylist_keypoints.get(i).getX();
            list_keypoints[i][1] = arraylist_keypoints.get(i).getY();
            list_keypoints[i][2] = arraylist_keypoints.get(i).getConfidence();
        }
        return list_keypoints;
    }

    public float[][] applyFilter() {
        float[][] arr_coordinates_with_confidence = new float[this.arraylist_all_frames_keypoints.get(0).size()][3];
        float[] list_proportional_weights = get_weights_for_filter(this.num_frames_for_filter - this.num_lowest_removed);
        //initialize sorted arraylist
        ArrayList<Keypoint>[] arraylist_sorted_keypoints = new ArrayList[this.arraylist_all_frames_keypoints.get(0).size()];
        for (int i = 0 ; i < arraylist_sorted_keypoints.length ; i++) {
            arraylist_sorted_keypoints[i] = new ArrayList<>();
        }
        //sort per keypoints
        for(int i = 0 ; i < this.arraylist_all_frames_keypoints.size() ; i++) {
            //System.out.println("this.arraylist_all_frames_keypoints.get(i).size() : " + this.arraylist_all_frames_keypoints.get(i).size());
            for(int j = 0 ; j < this.arraylist_all_frames_keypoints.get(i).size() ; j++) {
                arraylist_sorted_keypoints[j].add(this.arraylist_all_frames_keypoints.get(i).get(j));
            }
        }
        arraylist_sorted_keypoints = remove_keypoints_with_lowest_confidences(arraylist_sorted_keypoints);
        //apply weights
        for (int i = 0 ; i < arraylist_sorted_keypoints.length ; i++) { //keypoint idx
            for (int j = 0 ; j < arraylist_sorted_keypoints[i].size() ; j++) {  //frame idx
                arr_coordinates_with_confidence[i][0] += arraylist_sorted_keypoints[i].get(j).getX() * list_proportional_weights[j];
                arr_coordinates_with_confidence[i][1] += arraylist_sorted_keypoints[i].get(j).getY() * list_proportional_weights[j];
                arr_coordinates_with_confidence[i][2] += arraylist_sorted_keypoints[i].get(j).getConfidence() * list_proportional_weights[j];
            }
        }
        return arr_coordinates_with_confidence;



    }

    public float[] get_weights_for_filter(int num_frames) {
        float a = 0.8f;                //bigger -> more stable
        float num_power = 1.5f;        //smaller -> more stable, bigger -> faster response
        float[] list_weights = new float[num_frames];
        for (int i = 0 ; i < num_frames ; i++) {
            list_weights[i] = a * (float) Math.pow(i, num_power);
            //list_weights[i] = 1.0f/num_frames;
        }
        return get_softmax(list_weights);
    }

    public float[] get_softmax(float[] x) {
        float max = 0.0f;
        float sum = 0.0f;
        for(int i = 0 ; i < x.length ; i++) {
            if (max < x[i]) {
                max = x[i];
            }
        }
        for(int i = 0 ; i < x.length ; i++) {
            x[i] = (float) Math.exp(x[i] - max);
            sum += x[i];
        }
        for(int i = 0 ; i < x.length ; i++) {
            x[i] /= sum;
        }
        return x;
    }

    public ArrayList<Keypoint>[] remove_keypoints_with_lowest_confidences(ArrayList<Keypoint>[] arraylist_sorted_keypoints) {
        for(int i = 0 ; i < this.num_lowest_removed ; i++) {
            for (int j = 0 ; j < arraylist_sorted_keypoints.length ; j++) {
                arraylist_sorted_keypoints[j].remove(get_lowest_idx(arraylist_sorted_keypoints[j]));
            }
        }
        return arraylist_sorted_keypoints;
    }

    public int get_lowest_idx(ArrayList<Keypoint> arraylist_keypoints) {
        float low = 100;
        int lowIdx = 0;
        for(int i = 0 ; i < arraylist_keypoints.size() ; i++) {
            if (low > arraylist_keypoints.get(i).getConfidence()) {
                low = arraylist_keypoints.get(i).getConfidence();
                lowIdx = i;
            }
        }
        return lowIdx;
    }





    public class Keypoint {
        private int x;
        private int y;
        private float confidence;

        public Keypoint(int x, int y, float confidence) {
            this.x = x;
            this.y = y;
            this.confidence = confidence;
        }

        public int getY() {
            return y;
        }

        public void setY(int y) {
            this.y = y;
        }

        public float getConfidence() {
            return confidence;
        }

        public void setConfidence(float confidence) {
            this.confidence = confidence;
        }

        public int getX() {
            return x;
        }

        public void setX(int x) {
            this.x = x;
        }
    }

}

