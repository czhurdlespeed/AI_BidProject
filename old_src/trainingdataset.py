from preprocess import Contractor
import os
import pandas as pd
import numpy as np

def createXtrainYtrain(proposals):
    X_data = []
    y_data = []
    for proposal, infolist in proposals.items():
        for infotuple in infolist:
            X_data.append(infotuple[1])
            y_data.append(infotuple[2])
    return np.array(X_data), np.array(y_data)

if __name__ == "__main__":
    ## Read CSVs
    pd.set_option('future.no_silent_downcasting', True)
    listofreadCSVbidtabs = []
    for root,dirs, files in os.walk("ga_csv"):
        dirs.sort()
        files.sort()
        for file in files:
            filepath = os.path.join(root, file)
            listofreadCSVbidtabs.append(pd.read_csv(filepath))
    proposals = dict() 
    for proposal in listofreadCSVbidtabs:
        contractor = Contractor(proposal)
        slimmeddf = contractor.necessarycolumns(contractor.rawdataframe)
        cleaneddf = contractor.clean_numeric_columns(slimmeddf, ["Quantity", "Unit Price", "Extension"])
        
        contractor.cleaneddataframe = cleaneddf
        
        contractor.combineDuplicateLineItems()
        
        contractor.mergeDFwithGADOTpayitems()
        
        contractor.combinescaledNumericSingleProposal()
        
        contractor.encodedDataFrame = contractor.encodeCategoricalColumns(contractor.proposalScaledNumeric)
        
        contractor.extractContractors()
        
        contractor.sortbidsDeterminePlaces()
        
        contractor.setDollarlabels()
        
        proposals = contractor.createVectorLabelPairs(proposals)
        
    newproposals = contractor.addContractors(proposals)
        
    X_data, y_data = createXtrainYtrain(newproposals)
    print("X_data")
    print(X_data)
    print("y_data")
    print(y_data)
    np.save('X_data.npy', X_data)
    np.save('y_data.npy', y_data)
    print("Data saved to X_data.npy and y_data.npy")
    
        
        