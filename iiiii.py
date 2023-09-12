import pandas as pd,numpy as np
new_products_data = [
    ["Ladoo", " Yellow",99,12,9],
    ["Chilli Powder","Red",54,10,365],
    ["Curd","White",24,40,4]
]
data = [
    ["Apple","Red",5,50,7],
    ["Cheese","Yellow",6,200,13],
    ["Milk","White",5,90,6],
    ["Chocolate","Brown",6,10,90],
    ["Chicken Masala","Brown",7,10,300],
    ["Turmeric Powder","Yellow",7,10,180],
    ["Gulab Jamun","Butter",10,40,9]
    
]

columns = ["Product Name","Color","Quantity","Price per Unit(INR)","Shelf Life (Days)"]
df_org = pd.DataFrame(data,columns=columns)
df_org.head()
def solve(df,k,df_org):
    for it in range(0,k):
        for i in range(len(df)):
            df.iloc[i] = df.iloc[i] - 1
            if(df.iloc[i]<=0):
                df.iloc[i]=np.nan
            
    return df


k=int(input("Enter the days:"))
df_org["Shelf Life (Days)"]=solve(df_org["Shelf Life (Days)"],k,df_org)
df_org=df_org.dropna()
if(k>=1):
                new_data=pd.DataFrame(new_products_data,columns=columns)
                df_org = pd.concat([df_org, new_data], ignore_index = True,axis=0)
# df['Quantity']=solve(df['Quantity'])
# df.head())
# re_order = (df["Shelf Life (Days)","Quantity"] <= 0)
# reorder_products = df[re_order] 
    



print(df_org)
    
    
 