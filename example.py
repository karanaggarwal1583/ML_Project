import optimizer as opt

optimizer=["MGWOO","MGWO","GWO"]

datasets=["Phishing2"]
        
NumOfRuns=1

params = {"PopulationSize": 5, "Iterations": 30}

opt.run(optimizer, datasets, NumOfRuns, params)
                  
         
        
