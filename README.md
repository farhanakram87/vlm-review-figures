Digital pathology VLM review figures
`plotcode.py` generates figures from the Excel extraction workbook for the digital pathology vision-language model review.
Run locally
Install Python 3.12, or use an existing Python 3.12 installation.
Download this repository with Code → Download ZIP, then extract it.
Put your copy of `VLM_Extracted_Data.xlsx` beside `plotcode.py`. The workbook must contain the `Extraction_Clean` worksheet. If you keep the workbook elsewhere, pass its path to `--extraction-workbook` in quotes.
Open a terminal in the folder containing `plotcode.py` and `requirements.txt`. Install the dependencies and run the script.
```bash
   python -m pip install -r requirements.txt
   python plotcode.py --extraction-workbook VLM_Extracted_Data.xlsx --output-dir figures
   ```
If your system uses `python3` instead of `python`, replace it in both commands. On Windows, you can also use `py -3.12`.
Open the `figures` folder to find the PNG and PDF outputs.
The script produces the study overview from `Extraction_Clean`. With benchmark data, it also produces the task distribution and benchmark heatmap. It produces the metric violin plots when at least two of ACC, F1, AUC, BLEU, and C-index each have at least three numeric observations.
