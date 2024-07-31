import pandas as pd
import re
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import os
from collections import Counter

class Contractor:
    def __init__(self, rawdataframe):
        self.GADOT_PAYITEM_INDEX = list(pd.read_csv("PayItemIndex_2021.csv")["ITEM NO."])
        self.rawdataframe = rawdataframe
        self.cleaneddataframe = pd.DataFrame()
        self.individualcontractorDFs = dict()
        self.itemsEachContractorinProposal = dict()
        self.encodedDataFrame = pd.DataFrame()
        self.finalbids = dict()
        self.proposalScaledNumeric = pd.DataFrame()
        self.contractordata = dict()
        self.dollarlabels = []
        self.placebids = dict()
        self.winningbidamount = 0
        self.contractor_encoder = None
        self.top_contractors = None
        self.proposalContractorBids = dict()
        self.proposal = None
        
    # Clean Columns
    def clean_numeric_columns(self, df, columns):
        def clean_string(s):
            if pd.isna(s):
                return np.nan
            if isinstance(s, (int, float)):
                return float(s)
            if isinstance(s, str):
                cleaned =  re.sub(r'[^\d.]', '', str(s))
                try: 
                    return float(cleaned)
                except ValueError:
                    return np.nan
            return np.nan

        for col in columns:
            if col in df.columns:
                df.loc[:, col] = df.loc[:, col].apply(clean_string)
                # Convert to float if possible
                df.loc[:, col] = pd.to_numeric(df.loc[:,col])
            else:
                print(f"Warning: Column '{col}' not found in the DataFrame.")

        return df

    def necessarycolumns(self, df):
        columnsforFeatureVector = ["Proposal","Item", "Quantity", "Vendor Name", "Unit Price", "Extension"]
        featurevectordataframe = df[columnsforFeatureVector]
        return featurevectordataframe

    def combineDuplicateLineItems(self):
        for contractor in self.cleaneddataframe['Vendor Name'].unique():
            df = self.cleaneddataframe[self.cleaneddataframe["Vendor Name"] == contractor].copy()
            df.dropna(subset=['Item'], inplace=True)
            # Group by 'Item' and aggregate
            aggregated = df.groupby(['Proposal', 'Item', 'Vendor Name']).agg({
                'Quantity': 'sum',
                'Unit Price': 'mean',  # Note: This is still using mean for Unit Price
                'Extension': 'sum'
            }).reset_index()
            
            # Sort the dataframe to ensure consistent order
            aggregated = aggregated.sort_values(['Proposal', 'Item']).reset_index(drop=True)
            
            self.individualcontractorDFs[contractor] = aggregated

    def mergeDFwithGADOTpayitems(self):
        for contractorname, contractor in self.individualcontractorDFs.items():
            GADOT_payitemsDF = pd.DataFrame({"Item": self.GADOT_PAYITEM_INDEX})
            # Remove items from contractor not in GADOT pay items
            contractor = contractor[contractor["Item"].isin(self.GADOT_PAYITEM_INDEX)]
            allPayItems = pd.merge(GADOT_payitemsDF, contractor, on="Item", how="left")
            # Fill NaN values for specific columns
            allPayItems["Proposal"] = allPayItems["Proposal"].fillna(contractor["Proposal"].iloc[0])
            allPayItems["Vendor Name"] = allPayItems["Vendor Name"].fillna(contractor["Vendor Name"].iloc[0])
            self.proposal = allPayItems["Proposal"].iloc[0]
            # Fill NaN values for numeric columns with 0
            numeric_columns = ["Quantity", "Unit Price", "Extension"]
            allPayItems[numeric_columns] = allPayItems[numeric_columns].fillna(0);
            self.itemsEachContractorinProposal[contractorname] = allPayItems


    def scaleNumericColumns(self, contractor_name: str):
        numeric_features = ['Quantity', 'Unit Price', 'Extension']
        categorical_features = ['Item', 'Vendor Name', 'Bid Tab Rankings']
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
            ])
        pipeline = Pipeline([('preprocessor', preprocessor)])
        
        feature_vector = pipeline.fit_transform(self.itemsEachContractorinProposal[contractor_name])
        return pd.DataFrame(feature_vector)

    def combinescaledNumericSingleProposal(self):
        for contractorname, dataframe in self.itemsEachContractorinProposal.items():
            self.finalbids[contractorname] = dataframe['Extension'].sum()
            feature_df = self.scaleNumericColumns(contractorname)
            scaledNumericDF = dataframe
            scaledNumericDF["Quantity"] = feature_df[0]
            scaledNumericDF["Unit Price"] = feature_df[1]
            scaledNumericDF["Extension"] = feature_df[2]
            #print(scaledNumericDF.head())
            self.proposalScaledNumeric = pd.concat([self.proposalScaledNumeric, scaledNumericDF], axis=0)

    def encodeCategoricalColumns(self, dataframe):
        numeric_features = ['Quantity', 'Unit Price', 'Extension']
        categorical_features = ['Item', 'Vendor Name', 'Bid Tab Rankings']
        item_encoder = LabelEncoder()
        dataframe['Item'] = item_encoder.fit_transform(dataframe['Item'])
        vendor_encoder = OneHotEncoder(sparse_output=False)
        preprocessor = ColumnTransformer(
            transformers=[
                ('vendor_encoder', vendor_encoder, ['Vendor Name']),
                ('num', 'passthrough', ['Item','Proposal', 'Quantity', 'Unit Price', 'Extension'])
            ])
        pipeline = Pipeline([('preprocessor', preprocessor)])
        encoded_data = pipeline.fit_transform(dataframe)

        fitted_vendor_encoder = pipeline.named_steps['preprocessor'].named_transformers_['vendor_encoder']
        vendor_feature_names = fitted_vendor_encoder.get_feature_names_out(['Vendor Name'])

        all_feature_names = list(vendor_feature_names) + ['Item','Proposal', 'Quantity', 'Unit Price', 'Extension']

        
        return pd.DataFrame(encoded_data, columns = all_feature_names)

    def extractContractors(self):
        for column in self.encodedDataFrame.columns:
            if "Vendor Name" in column:
                self.contractordata[column[12:]] =self.encodedDataFrame[self.encodedDataFrame[column] > 0].loc[:,["Quantity", "Unit Price"]]

    def sortbidsDeterminePlaces(self):
        sorted_bids = sorted(self.finalbids.items(), key=lambda x: x[1])
        self.placed_bids = {contractor: {'amount': amount, 'place': place + 1} 
               for place, (contractor, amount) in enumerate(sorted_bids)}
    
    def setDollarlabels(self):
        self.dollarlabels = list(self.finalbids.values())
        self.winningbidamount = min(self.dollarlabels)

    
    def createVectorLabelPairs(self, proposals: dict):
        proposal = self.encodedDataFrame['Proposal'].iloc[0]
        proposals[proposal] = []
        for contractorname, contractordataframe in self.contractordata.items():
            if self.placed_bids[contractorname]['place'] == 1:
                won = 1
            else:
                won = 0
            proposals[proposal].append((contractorname, contractordataframe.to_numpy().flatten(), (won, self.winningbidamount, self.finalbids[contractorname])))
        return proposals

    def addContractors(self, proposals):
        newproposals = dict()
        contractorsencountered = []
        for proposal, infolist in proposals.items():
            for infotuple in infolist:
                contractorsencountered.append(infotuple[0])
        contractor_counts = Counter(contractorsencountered) 
        self.top_contractors = set([contractor for contractor, _ in contractor_counts.most_common(100)])
        self.contractor_encoder = OneHotEncoder(sparse_output=False)
        self.contractor_encoder.fit(np.array(list(self.top_contractors) + ['Other']).reshape(-1, 1))
        for proposal, infolist in proposals.items():
            newproposals[proposal] = []
            for infotuple in infolist:
                contractor_to_encode = infotuple[0] if infotuple[0] in self.top_contractors else 'Other'
                contractor_encoded = self.contractor_encoder.transform([[contractor_to_encode]]).flatten()
                contractorname, featurevector, labels = infotuple
                bidamount = np.array([labels[2]]).flatten()
                combinedfeatures = np.concatenate([contractor_encoded, featurevector, bidamount])
                newinfotuple = (contractorname, combinedfeatures, labels[0:2])
                newproposals[proposal].append(newinfotuple)
        return newproposals