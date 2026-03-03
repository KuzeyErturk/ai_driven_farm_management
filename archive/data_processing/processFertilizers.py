import pandas as pd


fertilizer_data = {
    'Growing_Season': ['2003/2004', '2004/2005', '2005/2006', '2006/2007', '2007/2008', 
                       '2008/2009', '2009/2010', '2010/2011', '2011/2012', '2012/2013',
                       '2013/2014', '2014/2015', '2015/2016', '2016/2017', '2017/2018',
                       '2018/2019', '2019/2020', '2020/2021', '2021/2022', '2022/2023', '2023/2024'],
    'Harvest_Year': [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013,
                     2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
    'Total_Nitrogen_kg_ha': [152, 150, 147, 148, 144, 139, 145, 146, 144, 137,
                             146, 146, 142, 138, 142, 137, 121, 130, 118, 125, 121],
    'Total_Phosphate_kg_ha': [41, 40, 35, 34, 31, 23, 30, 29, 28, 28,
                              29, 29, 29, 30, 27, 26, 24, 22, 17, 17, 17],
    'Total_Potash_kg_ha': [55, 54, 49, 47, 43, 33, 38, 39, 37, 40,
                           39, 38, 39, 37, 35, 34, 29, 28, 24, 23, 22]
}

fertilizer_df = pd.DataFrame(fertilizer_data)

# Keep only columns needed for merge
fertilizer_df = fertilizer_df[['Harvest_Year', 'Total_Nitrogen_kg_ha', 
                                'Total_Phosphate_kg_ha', 'Total_Potash_kg_ha']]
fertilizer_df = fertilizer_df.rename(columns={'Harvest_Year': 'Year'})

print("\nFertilizer Data (matched to harvest year):")
print(fertilizer_df)

print("\nFertilizer Statistics:")
print(f"Years: {fertilizer_df['Year'].min()} - {fertilizer_df['Year'].max()}")
print(f"N range: {fertilizer_df['Total_Nitrogen_kg_ha'].min()} - {fertilizer_df['Total_Nitrogen_kg_ha'].max()} kg/ha")
print(f"P range: {fertilizer_df['Total_Phosphate_kg_ha'].min()} - {fertilizer_df['Total_Phosphate_kg_ha'].max()} kg/ha")
print(f"K range: {fertilizer_df['Total_Potash_kg_ha'].min()} - {fertilizer_df['Total_Potash_kg_ha'].max()} kg/ha")

fertilizer_df.to_csv('data/uk_fertilizer_2004_2024.csv', index=False)
print("\nSaved: data/uk_fertilizer_2004_2024.csv")
