DESCRIPTION:
This package contains the following code components:
	1. Three models:
		a. Model 1 - predicts number of crimes per borough using categorical variables
		b. Model 2 - predicts crimes for a given year using times series regression
		c. Model 3 -  predics crime in given radius using spatical clustering and k-means
	2. Three visualizations:
		a. Interactive Barplot - generates a summary of crime data for the target location
		b. Heatmaps - demonstrate high crime density in dark color on map
		c. Final interactive Visualization - the final interactive visualization based on danger scores calculated using model 2 results. 
and the following documents:
	1. final report
	2. final poster
	
	
	
INSTALLATION:
	1. Python
	2. Tableau
	3. Download the NYC crime data from https://data.cityofnewyork.us/Public-Safety/NYC-crime/qb7u-rbmr
	
	
	
EXECUTION:	
	1. To run model 1:
		a. Open Model 1 folder and open Model 1.ipynb
		b. Run the model
	
	2. To run model 2:
		a. Follow the instructions outlined in:
		https://github.gatech.edu/jrich34/CSE-6242/tree/main/Crime%20Prediction%20Demo
	
	3. To run model 3:
		a. Open Model 3 folder and open Clustering.ipynb
		b. import data and run the model	

	4. To generate the interactive barplot
		a. Open Interactive Visualization folder
		b. Open newyorkdata.twb 
		c  Reconnect data source using the downloaded csv data
		d. click on interactive barplot tab. 
		e. In the input box on the right side, enter
			- latitude of target location 
			- longitude of target location
			- radius

	5. To generate the heatmap:
		a. Open Heatmap folder
		b. Only Tableau is used to generate the heatmap (openrefine was used to pre-process the data)
		c. Open the .csv file for a certain year
		d. Use "Longitude" as Columns, "Latitude" as Rows, "Cmplnt" as Details
		e. Under "marks", change the drop-down list from "Automatic" to "Density"
		f. In "Color", change the color, set intensity to 80% for best visualization
	
	6. To generate the final interactive visualization	
		a. Open Interactive Visualization folder
		b. Open Inreractive Danger Score Map.twb
		c. Reconnect data source using data generated from model 2
		d. Go to Interactive Crime Map tab
		e. Hover mouse over NTA to see longitude, latitude, NTA name and danger score
