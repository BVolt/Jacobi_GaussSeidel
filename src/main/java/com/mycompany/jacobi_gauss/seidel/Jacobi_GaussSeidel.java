
package com.mycompany.jacobi_gauss.seidel;

import java.io.*;
import java.util.*;



public class Jacobi_GaussSeidel {
    private ArrayList<double[]> matrix = new ArrayList<>();
    private final int rowcnt, colcnt, rowIndex, colIndex ;
    private static BufferedReader reader;
    private double[] currResult;
    private double[] prevResult;
    private final double targetError;
    private int iteration = 0;
    private static Scanner input;
    
    public static void main(String[] args) {
        //Variable Initialization
        input = new Scanner(System.in);
        int n = getN();
        Jacobi_GaussSeidel proj2 = new Jacobi_GaussSeidel(n, getMatrix(n), getError(), getStartingVals(n));
        proj2.printMatrix();
        if(proj2.diagDom()){
            System.out.println("Coefficient Matrix is Diagonally Dominant");
            proj2.jacobi();
            proj2.gauss_seidel();  
        }
        else{
            System.out.println("Coefficient Matrix is not Diagonally Dominant!\nValues may not be accurate");
            System.exit(0);
        }
    }

    public Jacobi_GaussSeidel(int n, ArrayList<double[]> matrix, double targetError, double[] startingVals){
        this.rowcnt = n;
        this.colcnt = n+1;
        this.rowIndex = rowcnt - 1;
        this.colIndex = colcnt - 1;
        this.currResult = new double[n];
        this.prevResult = startingVals;
        this.targetError = targetError;
        this.matrix = matrix;
    }

    public void gauss_seidel() {
        while (errorLimitExceeded() && iteration < 50) {
            System.arraycopy(currResult, 0, prevResult, 0, currResult.length);
            double runningTotal;
            boolean[] writtenVar = new boolean[rowcnt];
            for (int i = 0; i < rowcnt; i++) {
                runningTotal = matrix.get(i)[colIndex];
                for (int j = 0; j < colIndex; j++) {
                    double varVal;
                    if (writtenVar[j]) 
                        varVal = currResult[j];
                    else 
                        varVal = prevResult[j];
                    if (i == j) continue;
                    runningTotal -= (matrix.get(i)[j] * varVal);
                }
                currResult[i] = (runningTotal / matrix.get(i)[i]);
                writtenVar[i] = true;
            }
            iteration++;
            printResult("Gauss-Seidel");
            System.out.printf("L2 Norm: %f\n",calcL2(currResult));
        }
        if(iteration ==50)System.out.println("Error Limit not reached");
        iteration = 0;
        prevResult = new double[rowcnt];
        currResult = new double[rowcnt];
    }
    
    public void jacobi() {
        while (errorLimitExceeded() && iteration < 50) {
            System.arraycopy(currResult, 0, prevResult, 0, currResult.length);
            double runningTotal;
            for (int i = 0; i < rowcnt; i++) {
                runningTotal = matrix.get(i)[colIndex];
                for (int j = 0; j < colIndex; j++) {
                    if (i == j) continue;
                    runningTotal -= (matrix.get(i)[j] * prevResult[j]);
                }
                currResult[i] = (runningTotal / matrix.get(i)[i]);
            }
            iteration++;
            printResult("Jacobi");
            System.out.printf("L2 Norm: %f\n",calcL2(currResult));
        }
        if(iteration ==50)System.out.println("Error Limit not reached");
        iteration = 0;
        prevResult = new double[rowcnt];
        currResult = new double[rowcnt];
    }

    public boolean errorLimitExceeded(){
        if(iteration ==0) return true;
        double error = Math.abs(calcL2(currResult) - calcL2(prevResult));
        System.out.printf("Relative Error | L2Curr - L2Prev |: %f\n", error);
        return targetError < error;
    }
    
    public boolean diagDom(){
        for(int i = 0; i < rowcnt; i++){
            double total = 0;
            for(int j = 0; j< colIndex; j++){
                if(i==j) continue;
                total += Math.abs(matrix.get(i)[j]);
            }
            if(matrix.get(i)[i] < total)
                return false;
        }
        return true;
    }
    
    public double calcL2(double[] xVector){
        double l2 = 0;
        double sqSum = 0;
        for(int i = 0; i<rowcnt; i++){
            sqSum += Math.pow(xVector[i], 2);
        } 
        l2 = Math.sqrt(sqSum);        
        return l2;
    }
    
    public void printResult(String methodName){
        System.out.println("=========================================================");
        System.out.printf("%s method after %d iterations\n", methodName, iteration);
        System.out.println("=========================================================");
        System.out.printf("X Vector %d: ", iteration);
        System.out.println(Arrays.toString(currResult));
    }
    
    public void printMatrix(){
        System.out.println("=========================================================");
        System.out.println("Input Matrix");
        System.out.println("=========================================================");
        for(int i = 0; i < matrix.size(); i++){
            System.out.print("{");
            for(int j = 0; j < colcnt; j++){
                System.out.print(matrix.get(i)[j]);
                if(j != rowcnt)
                    System.out.print(", ");
            }
            System.out.print("}\n");
        }
    }
    
    public static int getN(){
        int n=0;
        //Get number of linear equations
        while(n <=0 || n >10){
            if((n <0 || n >10)) System.out.println("Invalid value for n!");
            System.out.print("Please provide number (n) of linear equations you would like to solve (0 < n <= 10): ");
            n = input.nextInt();
        }
        return n;
    }
    
    public static double[] getStartingVals(int n){
        //Get starting solutions
        double[] startingVals = new double[n];
        System.out.println("Please provide the starting solution for iterative methods");
        for(int i = 0; i < startingVals.length; i++){
            System.out.printf("Value of X%d: ", i);
            startingVals[i] = input.nextDouble();
        }
        
        return startingVals;
    }
    
    public static ArrayList<double[]> getMatrix(int n) {
        System.out.print("""
                         1: Read in a File
                         2: Manually enter coefficients
                         Enter 1 or 2:  """);
        int choice = input.nextInt();
        if (choice == 1) return readFile(n);
        else return inputMatrix(n);
    }
    
    public static ArrayList<double[]> readFile(int n){
        Scanner inp = new Scanner(System.in);
        System.out.print("Please provide filename: ");
        String fileName = inp.nextLine();        
        ArrayList<double[]> matrix = new ArrayList<>();
        try{
            reader = new BufferedReader(new FileReader(fileName));
            String line;
            while((line = reader.readLine()) != null){
                String[] splitLine = line.split(" ");
                double[] row = new double[n+1];
                
                for(int i = 0; i < splitLine.length; i++){
                    row[i] = Double.parseDouble(splitLine[i]);
                }
                
                matrix.add(row);
            }
        }catch(IOException e){
        
        }
        return matrix;
    }

    public static ArrayList<double[]> inputMatrix(int n) {
        ArrayList<double[]> matrix = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            System.out.printf("\nEquation %d\n", i + 1);
            double[] array = new double[n+1];
            for (int j = 0; j < n+1; j++) {
                if (j == n) {
                    System.out.print("Please enter equation solution: ");
                } else {
                    System.out.printf("Please enter coefficient for X%d: ", j + 1);
                }
                array[j] = input.nextDouble();
            }
            matrix.add(array);
        }
        
        return matrix;
    }

    public static double getError(){
        //Get relative error
        System.out.print("Please provide error threshold to terminate calculations: ");
        return input.nextDouble();
    }
}
