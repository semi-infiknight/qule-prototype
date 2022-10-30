package com.Qule.FileUploadpkg;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

import net.sf.jasperreports.engine.JRException;
import net.sf.jasperreports.engine.JasperCompileManager;
import net.sf.jasperreports.engine.JasperFillManager;
import net.sf.jasperreports.engine.JasperPrint;
import net.sf.jasperreports.engine.JasperReport;
import net.sf.jasperreports.engine.data.JRBeanCollectionDataSource;
import net.sf.jasperreports.engine.export.JRPdfExporter;
import net.sf.jasperreports.engine.util.JRSaver;
import net.sf.jasperreports.export.SimpleExporterInput;
import net.sf.jasperreports.export.SimpleOutputStreamExporterOutput;
import net.sf.jasperreports.export.SimplePdfExporterConfiguration;
import net.sf.jasperreports.export.SimplePdfReportConfiguration;

public class FileReading {
	
//	public static void main(String[] args)
//	{
//		pdfGeneration();
//		//sdfFileReading("C:\\Users\\Tarush\\hackathon\\Qule\\Qule\\src\\main\\java\\com\\Qule\\FileUploadpkg\\gdb9.sdf.csv");
//	}

	private List<List<String>> propertyReading(String fileName)
	{		
		File file= new File(fileName);

        // this gives you a 2-dimensional array of strings
        List<List<String>> lines = new ArrayList<>();
        Scanner inputStream;

        try{
            inputStream = new Scanner(file);

            while(inputStream.hasNext()){
                String line= inputStream.next();
                String[] values = line.split(",");
                // this adds the currently parsed line to the 2-dimensional string array
                lines.add(Arrays.asList(values));
            }

            inputStream.close();
        }catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        int lineNo = 1;
        for(List<String> line: lines) {
            int columnNo = 1;
            for (String value: line) {
                System.out.println("Line " + lineNo + " Column " + columnNo + ": " + value);
                columnNo++;
            }
            lineNo++;
        }
        return lines;
	}
	
	public void pdfGeneration()
	{
		try {
			//Value for data source
			//"C:\\Users\\Tarush\\hackathon\\Qule\\Qule\\src\\main\\java\\com\\Qule\\FileUploadpkg\\Test.csv"
			//"C:\\Users\\Tarush\\hackathon\\Qule\\Qule\\src\\main\\java\\com\\Qule\\FileUploadpkg\\MoleculeReport.jrxml"
			String csvPath=System.getProperty("user.dir")+File.separator+"CSV_FILE"+File.separator+"molmetrics.csv";
			String jrxmlPath=System.getProperty("user.dir")+File.separator+"REPORT_XML"+File.separator+"MoleculeReport.jrxml";
			List<List<String>> RC = propertyReading(csvPath);
			List<MolProperty> MPList = new ArrayList<>();
			for (int i = 0; i < RC.size(); i++) {
				MolProperty MP = new MolProperty();
				//Bug in the code
				for (int j = 0; j < RC.get(i).size(); j+=4) 
				{
					MP.setNatural_product_scores(RC.get(i).get(j));
					MP.setQuantitative_estimation_druglikeness_scores(RC.get(i).get(j+1));
					MP.setSynthetic_accessibility_score_scores(RC.get(i).get(j+2));
					MP.setWater_octanol_partition_coefficient_scores(RC.get(i).get(j+3));
				}
				MPList.add(MP);
			}
			
			//Parameter
			Map<String, Object> parameters = new HashMap<>();
			parameters.put("title", "Molecule Report");
			//parameters.put("Sybil",ClassLoader.getSystemResourceAsStream("Sybil.png"));
			
			//InputStream moleculeReport= getClass().getResourceAsStream("C:\\Users\\Tarush\\hackathon\\Qule\\Qule\\src\\main\\java\\com\\Qule\\FileUploadpkg\\MoleculeReport.jrxml");
			JRBeanCollectionDataSource dataSource = new JRBeanCollectionDataSource(MPList);
			//JasperReport JR = JasperCompileManager.compileReport(moleculeReport);
			JasperReport JR = JasperCompileManager.compileReport(jrxmlPath);
			JRSaver.saveObject(JR, "MoleculeReport.jasper");
			JasperPrint jasperPrint = JasperFillManager.fillReport(JR,parameters,dataSource);
			
			JRPdfExporter exporter = new JRPdfExporter();

			exporter.setExporterInput(new SimpleExporterInput(jasperPrint));
			exporter.setExporterOutput(new SimpleOutputStreamExporterOutput("MoleculeReport.pdf"));

			SimplePdfReportConfiguration reportConfig = new SimplePdfReportConfiguration();
			reportConfig.setSizePageToContent(true);
			reportConfig.setForceLineBreakPolicy(false);

			SimplePdfExporterConfiguration exportConfig = new SimplePdfExporterConfiguration();
			exportConfig.setMetadataAuthor("QULE");
			exportConfig.setEncrypted(true);
			exportConfig.setAllowedPermissionsHint("PRINTING");

			exporter.setConfiguration(reportConfig);
			exporter.setConfiguration(exportConfig);

			exporter.exportReport();
			System.out.println("Report Printed");
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	public List<MolProperty> CSVReading()
	{
		String csvPath=System.getProperty("user.dir")+File.separator+"CSV_FILE"+File.separator+"molmetrics.csv";
		String line = "";  
		String splitBy = ",";
		List<MolProperty> molList = new ArrayList<>();
		try   
		{    
		BufferedReader br = new BufferedReader(new FileReader(csvPath));  
		while ((line = br.readLine()) != null)   //returns a Boolean value  
		{
			MolProperty MolProp = new MolProperty();	
			String[] employee = line.split(splitBy);
			MolProp.setNatural_product_scores(employee[0]);
			MolProp.setQuantitative_estimation_druglikeness_scores(employee[1]);
			MolProp.setSynthetic_accessibility_score_scores(employee[2]);
			MolProp.setWater_octanol_partition_coefficient_scores(employee[3]);
			molList.add(MolProp);
		}
		}catch(Exception e)
		{
			e.printStackTrace();
		}
		return molList;
	}
	}
