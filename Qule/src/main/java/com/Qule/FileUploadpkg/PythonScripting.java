package com.Qule.FileUploadpkg;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;


public class PythonScripting {

	public PythonScripting() {
		// TODO Auto-generated constructor stub
	}
	
//	public static void main(String[] args)
//	{
//		try {
//			pythonScriptRun();
//		} catch (IOException | InterruptedException e) {
//			// TODO Auto-generated catch block
//			e.printStackTrace();
//		}
//	}
	
	public void pythonScriptRun() throws IOException, InterruptedException
	{
		Process process;
		boolean isWindows = System.getProperty("os.name").toLowerCase().startsWith("windows");
		if(!isWindows)
		{
			System.exit(0);
		}
		//==================================================================================================
		String pythonFilePath=System.getProperty("user.dir")+File.separator+"MLAPI"+File.separator+"qule.py";
		String absoluteQulePath="C:/Users/Tarush/anaconda3/envs/quleapp/python";
	  	ProcessBuilder processBuilder = new ProcessBuilder(absoluteQulePath, pythonFilePath);
	    processBuilder.redirectErrorStream(true);
	    process = processBuilder.start();
	    BufferedReader in = new BufferedReader(new InputStreamReader(process.getInputStream()));
	    BufferedReader error = new BufferedReader(new InputStreamReader(process.getErrorStream()));
	    String line=null;
	    while((line=in.readLine())!=null)
	    {
	    	System.out.println("Python Code :: "+line);
	    }
	    while((line=error.readLine())!=null)
	    {
	    	System.out.println("Error :: "+line);
	    }
	    int exitCode = process.waitFor();
	    System.out.println(exitCode);
	    //Fetching the values from python in java
	    
	    //
	}
}
