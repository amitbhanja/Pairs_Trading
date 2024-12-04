# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 15:20:21 2024

@author: Amit Bhanja
"""

import itertools
from Pairs import Pairs
import multiprocessing as mp
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import matplotlib.pyplot as plt

maxHalflife = 60
nyse_stocks = {
    "Banking": [
        "JPM",  # JPMorgan Chase & Co.
        "BAC",  # Bank of America Corp.
        "WFC",  # Wells Fargo & Co.
        "C",    # Citigroup Inc.
        "GS",   # Goldman Sachs Group Inc.
        "MS",   # Morgan Stanley
        "USB",  # U.S. Bancorp
        "TFC",  # Truist Financial Corporation
        "PNC",  # PNC Financial Services Group Inc.
        "KEY",  # KeyCorp
        "RF",   # Regions Financial Corporation
        "MTB",  # M&T Bank Corporation
        "FITB", # Fifth Third Bancorp
        "HBAN", # Huntington Bancshares Incorporated
        "FRC"   # First Republic Bank
    ],
    "IT": [
        "IBM",  # International Business Machines Corporation
        "ORCL", # Oracle Corporation
        "CRM",  # Salesforce.com Inc.
        "CSCO", # Cisco Systems, Inc.
        "DELL", # Dell Technologies Inc.
        "ACN",  # Accenture plc
        "DXC",  # DXC Technology Co.
        "HPE",  # Hewlett Packard Enterprise Co.
        "AMD",  # Advanced Micro Devices, Inc.
        "STX",  # Seagate Technology Holdings PLC
        "MU",   # Micron Technology, Inc.
        "WDC",  # Western Digital Corporation
        "ANET", # Arista Networks, Inc.
        "CTSH", # Cognizant Technology Solutions Corp.
        "ADBE"  # Adobe Inc.
    ],
    "Pharma": [
        "PFE",  # Pfizer Inc.
        "JNJ",  # Johnson & Johnson
        "MRK",  # Merck & Co., Inc.
        "ABBV", # AbbVie Inc.
        "BMY",  # Bristol-Myers Squibb Company
        "LLY",  # Eli Lilly and Company
        "GILD", # Gilead Sciences, Inc.
        "AMGN", # Amgen Inc.
        "REGN", # Regeneron Pharmaceuticals, Inc.
        "ZTS",  # Zoetis Inc.
        "BIIB", # Biogen Inc.
        "VRTX", # Vertex Pharmaceuticals Incorporated
        "ALXN", # Alexion Pharmaceuticals, Inc.
        "MRNA", # Moderna, Inc.
        "NVS"   # Novartis AG
    ],
    "Energy": [
        "XOM",  # Exxon Mobil Corporation
        "CVX",  # Chevron Corporation
        "COP",  # ConocoPhillips
        "PSX",  # Phillips 66
        "SLB",  # Schlumberger Limited
        "OXY",  # Occidental Petroleum Corporation
        "HAL",  # Halliburton Company
        "MRO",  # Marathon Oil Corporation
        "VLO",  # Valero Energy Corporation
        "BKR",  # Baker Hughes Company
        "HES",  # Hess Corporation
        "KMI",  # Kinder Morgan, Inc.
        "EOG",  # EOG Resources, Inc.
        "PXD",  # Pioneer Natural Resources Company
        "FANG"  # Diamondback Energy, Inc.
    ],
    "Consumer Goods": [
        "PG",   # Procter & Gamble Co.
        "KO",   # The Coca-Cola Company
        "PEP",  # PepsiCo, Inc.
        "MO",   # Altria Group, Inc.
        "PM",   # Philip Morris International Inc.
        "CL",   # Colgate-Palmolive Company
        "KMB",  # Kimberly-Clark Corporation
        "NKE",  # Nike, Inc.
        "MNST", # Monster Beverage Corporation
        "TAP",  # Molson Coors Beverage Company
        "CLX",  # The Clorox Company
        "GIS",  # General Mills, Inc.
        "HSY",  # The Hershey Company
        "K",    # Kellogg Company
        "CHD"   # Church & Dwight Co., Inc.
    ]
}

def performPairsTradingStrategy(stock_pair):
    findings = []
    stock1 = stock_pair[0]
    stock2 = stock_pair[1]
    pairs = [Pairs(stock1, stock2, '2012-01-01', '2020-12-31'), Pairs(stock2, stock1, '2012-01-01', '2020-12-31')]
    for pair in pairs:
        pair.collect_data()
        if pair.findOptimumSpread(90):
            half_life = pair.half_life()
            if half_life <= 0 or half_life > maxHalflife:
                print(f"Half Life for stocks {stock1} and {stock2} : {half_life} days")
            else:
                ret = pair.mean_reversion_strategy()
                findings.append([(pair.stock1, pair.stock2), (ret[0], ret[3], ret[2], ret[1], ret[4], ret[5].copy())])
        else:
            print(f"Cannot find optimum spread between {stock1} and {stock2}")
    return findings
    

def write_strategy_findings_pdf(findings, filename='Pairs_Trading_Finding.pdf'):
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, height - 40, "Stock Pairs Trading Report")
    
    c.setFont("Helvetica", 12)
    
    y_position = height - 80
    print(f'Findings {len(findings)}')
    for finding in findings:
        #print(finding)
        if len(finding) == 0:
            continue
        #pair, metrics = finding
        for result in finding:
            stock1, stock2 = result[0]
            cagr, max_drawdown, success_ratio, std_dev, hedge_ratio, df = result[1]
            
            # Write the results for each stock pair
            c.drawString(100, y_position, f"Pair: {stock1} & {stock2}")
            y_position -= 20
            c.drawString(120, y_position, f"CAGR: {cagr:.2%}")
            y_position -= 20
            c.drawString(120, y_position, f"Max Drawdown: {max_drawdown:.2%}")
            y_position -= 20
            c.drawString(120, y_position, f"Success Ratio: {success_ratio:.2%}")
            y_position -= 20
            c.drawString(120, y_position, f"Std Deviation: {std_dev:.2%}")
            y_position -= 40
            c.drawString(120, y_position, f"Equation: {stock1} - {hedge_ratio} * {stock2}")
            y_position -= 40
    
            plt.figure(figsize=(8, 4))
            df['cum_returns'].plot(label='Cumulative Returns', color='magenta')
            plt.xlabel('Date')
            plt.ylabel('Cumulative Returns')
            plt.legend()
            plt.tight_layout()
            image_path = f"cumulative_returns_{stock1}_{stock2}.png"
            plt.savefig(image_path)
            plt.close()
    
            c.drawImage(image_path, 100, y_position - 150, width=400, height=150)
            y_position -= 200
    
            # Move to new page
            if y_position < 100:
                c.showPage()
                y_position = height - 80
    
    c.save()

def main():
    stock_pairs = {}
    for sector, stocks in nyse_stocks.items():
        stock_pairs[sector] = list(itertools.combinations(stocks, 2))
    
    complete_results = []
    for sector in stock_pairs.keys():
        with mp.Pool(processes=4) as pool:
            findings = pool.imap_unordered(performPairsTradingStrategy, stock_pairs[sector])
            complete_results.extend(findings)
            
    write_strategy_findings_pdf(complete_results)
    
if __name__ == "__main__":
    main()