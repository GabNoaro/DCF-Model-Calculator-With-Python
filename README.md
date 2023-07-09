# Projects-with-Python

This scripts allow you to upload financial statements in .csv format (e.g., downloaded from Morningstar.com), and automatically estimate the fair price of the company based on a discounted cash flow model. The fair price depend on different variables, such as the WACC you decide (which you can change in the formula). Moreover, the compounded growth of each financial value is calculated based on a weighted average of the last few years, that gives bigger weights to certin year (e.g., in this example it gives smaller weights to the COVID-19 years, because they caused anomalies in the economy.

This script is by no means a predictor and it isn't a magic formula for financial success, it's up to you to change the weights in the weighted average YOY growth, to estimate the WACC, to interpret the results, and to spot any bug in the code. If you find any bug in the code, please let me know, I'll be happy to fix it and improve the code. Moreover, it's impossible to predict the future, and you should be cautious when interpreting the fair price and any other result calculated by this script.

This script uses Roche Holding AG (RHHBY) as an example, and you may have to change the attribute names when importing your own three financial statements, because those attributed may be named differently (e.g., Total Revenue could be named Revenue or have other names).
