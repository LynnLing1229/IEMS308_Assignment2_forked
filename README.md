# IEMS Assignment 2
Through analyzing the transaction data of Dillard's, suggest SKUs that are the best candidates to modify the planograms. 

## How to run 
* Because it takes a long time to run the code, I do not really recommend running the code. I kept Notebook Output for you. I also attachedk the HTML file from Jupyter Notebook.
But if you really want to run it, you can follow the following instructions. 
* Download [trnsact.csv](https://drive.google.com/file/d/16k7guGlvVxp-n1h5y6HzGGytGir1sL6E/view?usp=sharing), [trnsact_labelled.csv](https://drive.google.com/file/d/1ZJYQSsqe9C9-Rn9qYhuymCCHRA134U5a/view?usp=sharing), [trnsact_labelled_filtered0.csv](https://drive.google.com/file/d/1UwkMf0GtgabFf9Jijy7kxU3ikyyRjy1j/view?usp=sharing), and [combination of five files(skstinfo, skuinfo, skstinfo_labelled, skuinfo_labelled, trnsact_labelled_filtered1)](https://drive.google.com/file/d/1HVW8_NMF4IEHIMNuV7Av2ML22GtfGWaR/view?usp=sharing). Put those file in "data/Dillards POS/" folder. I could not push those files because the file size of those files exceeded the github limit. 
* Download [ForApriori.csv](https://drive.google.com/file/d/1iU0jmqSCkS3RyA7n2wj5VQySl2vZMTay/view?usp=sharing). Put that file in "data/" folder. I could not push those files because the file size of those files exceeded the github limit. 
* Open source_code.ipynb file and run each code snippet. I recommend this way than running source_code.py
* Or run source_code.py.

### Prerequisites
* Python (numpy, pandas, csv, copy, mlxtend, matplotlib, random)

### What file to check for what
On Canvas announcement, it says that we need to submit 4 things. 
* The source code: check "source_code.ipynb" or "source_code.html" or "source_code.py". I recommend "source_code.ipynb" and "source_code.html" because they are easier to read and because they have outputs. 
* Sample output of the code: Check "data/which_SKU_to_WHERE.csv" or "data/AssocRules.csv".
* A half a page 'executive summary' of the findings: Check "Executive Summary.pdf"
* Document with all findings: Check "Report.pdf" 

## File Structure
```
IEMS308_Assignment2
├── README.md 						: This document.
├── source_code.ipynb 					: Code for this assignment. Recommend openning this rather than py file. Recommended
├── source_code.html					: Same as source code.ipynb but in HTML format. Recommended
├── source_code.py 					: Same as source code.ipynb but in Python format. Not Recommended
├── Report.pdf 						: Report on this assignment 2. This also includes Executive Summary.
├── Executive Summary.pdf 				: Copy of Executive Summary of the report.
├── data
│	├── Dillards POS
│	│	├── trnsact.csv 			: Original data file. Download it as explained in "How to run" section.
│	│	├── skstinfo.csv		: Original data file. Download it as explained in "How to run" section.
│	│	├── strinfo.csv				: Original data file.
│	│	├── skuinfo.csv				: Original data file. Download it as explained in "How to run" section.
│	│	├── deptinfo.csv			: Original data file.
│	│	├── trnsact_labelled.csv 		: Original data file with labels. Download it as explained in "How to run" section.
│	│	├── skstinfo_labelled.csv	: Original data file with labels. Download it as explained in "How to run" section.
│	│	├── strinfo_labelled.csv				: Original data file with labels.
│	│	├── skuinfo_labelled.csv					: Original data file with labels. Download it as explained in "How to run" section.
│	│	├── deptinfo_labelled.csv					: Original data file with labels.
│	│	├── trnsact_labelled_filtered0.csv	: Intermediate filtered trnsact data. Download it as explained in "How to run" section.
│	│	├── trnsact_labelled_filtered1.csv				: Intermediate filtered trnsact data. Download it as explained in "How to run" section.
│	│	├── trnsact_labelled_filtered.csv				: Final filtered trnsact data.
│	│	├── strinfo_labelled_filtered.csv					: Filtered strinfo data
│	│	├── skstinfo_labelled_filtered.csv					: Filtered skstinfo data
│   │   └── sku_deptinfo_labelled_filtered.csv 				: Filtered skuinfo and deptinfo data (combined)
│   ├── AssocRules.csv 					:  Association rules filtered based on minsup, minconf, and minlift
│   ├── ForApriori.csv 					:  Data that gets fed to create AssocRules. Download it as explained in "How to run" section.
│   ├── FreqItems.csv 					:  List of frequent items based on minsup
│   └── which_SKU_to_WHERE.csv 			:  About what SKU to move to where 
└── img
	├── return rate.png 				: Step 1.3.1.
	├── return rate  (only 0% to 100%).png	: Step 1.3.1.
	├── Distribution of number of unique SKUs that each store has.png    : Step 1.3.3.
	├── Number of transactions per day.png		: Step 1.3.2.
	└── How much does each STORE support my proposed SKU rearrangement plan?.png	: Step 4.4.
```

## Author
JunHwa Lee

