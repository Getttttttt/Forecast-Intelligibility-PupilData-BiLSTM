if __name__ == "__main__":
    sys.stdout = open('./Record/outputEnComprehensibility.txt', 'w')
    dataframe = pd.read_csv('./Dataset/EnData.csv', low_memory=False)
    
    print(dataframe.head(10))