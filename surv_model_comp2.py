    #!/usr/bin/env python3
    # -*- coding: utf-8 -*-

    # %% Libraries and read of data
    from sksurv.nonparametric import kaplan_meier_estimator
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from sksurv.metrics import concordance_index_censored
    from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
    from sksurv.svm import FastSurvivalSVM, FastKernelSurvivalSVM
    from sklearn.model_selection import GridSearchCV, ShuffleSplit
    from sklearn import set_config
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import random
    import warnings
    from joblib import dump
    warnings.filterwarnings("ignore")

    wc_file = r'C:\Users\marga\Desktop\new_data_comparacao\machine1_data_final.csv'
    x_features = ['volt', 'rotate', 'pressure', 'vibration', 'cycle_comp2']
    df_wc = pd.read_csv(r'C:\Users\marga\Desktop\new_data_comparacao\machine1_data_final.csv')
    df_wc['datetime'] = pd.to_datetime(df_wc['datetime'], format='%d/%m/%Y %H:%M')

    # convert replace to status col
    df_wc['status'] = df_wc['replace_comp2'].map({True: False, False: True})
    df_wc['ttf_comp2'] = df_wc['ttf_comp2'] + 1


    # %% downsamples the data to balance the classes
    df_major = df_wc[df_wc['replace_comp2'] == False]               # não ocorre manutençoes
    df_minor = df_wc[df_wc['replace_comp2'] == True]                # manutençoes que acontecem
    df_major_sample = df_major.sample(frac=0.1, random_state=22)    # Downsampling: It then takes a random sample (10%) of the df_major subset to balance the classes, ensuring there's not an overwhelming majority of one class over the other. 
    df_sample = pd.concat([df_major_sample, df_minor])
    df_sample.sort_index(inplace=True)

   
    # %% plots the survival function
    time, survival_prob = kaplan_meier_estimator(df_sample["status"], df_sample["ttf_comp2"])
    plt.step(time, survival_prob, where="post")
    plt.ylabel("est. probability of survival $\hat{S}(t)$")
    plt.xlabel("cycles $c$")

    # %% splits the dataset
    def dataset_split(df, sample2test):
        # gets the limits
        idx_end = df[df['replace_comp2'] == True].index[sample2test]
        idx_start = df[df['replace_comp2'] == True].index[sample2test - 1] if sample2test != 0 else 0
        # computes the slices
        test_idx = (df.index > idx_start) & (df.index <= idx_end)
        x_train = df.loc[~test_idx, x_features].values
        y_train = np.array(list(zip(df.loc[~test_idx, 'status'], df.loc[~test_idx, 'ttf_comp2'])),
                        dtype=[('status', '?'), ('ttf_comp2', '<f8')])
        x_test = df.loc[test_idx, x_features].values
        y_test = np.array(list(zip(df.loc[test_idx, 'status'], df.loc[test_idx, 'ttf_comp2'])),
                        dtype=[('status', '?'), ('ttf_comp2', '<f8')])
        return x_train, y_train, x_test, y_test
    # splits the data
    x_train, y_train, x_test, y_test = dataset_split(df_sample, 3) 


        
    # %% plot survival function
    def plot_survival_function(model, x_test, y_test, idx2test):
        pred_surv = model.predict_survival_function(x_test[idx2test])
        time_points = np.arange(1, 2161)
        for i, surv_func in enumerate(pred_surv):
            plt.step(time_points, surv_func(time_points), where="post",
                    label="c={0}, ttf_comp2={1}".format(round(x_test[idx2test[i], -1]), int(y_test[idx2test[i]][1])))
        plt.ylabel("est. probability of survival $\hat{S}(t)$")
        plt.xlabel("cycles $c$")
        plt.legend(loc="best")
        plt.title('Fig.1: Predicted survival function for cycles={0}'.format(x_test[idx2test, -1]))
        plt.show()

    def train_model(model):
        model.fit(x_train, y_train)
        # evaluates the model
        cindex = model.score(x_train, y_train)
        print('cindex train:', round(cindex, 3))
        cindex = model.score(x_test, y_test)
        print('cindex test:', round(cindex, 3))

    # %% fits the cox model
    cox_model = CoxPHSurvivalAnalysis()
    print('### COX Model ###')
    train_model(cox_model)
    # features importance
    print('feature importance:')
    print(pd.Series(cox_model.coef_, index=x_features))
    # evalutes the survival functions
    idx2test = [0, 48, 97]
    #idx2test = [50, 300, 650]
    #idx2test = np.random.choice(203, 3)  # Select 3 random indices
    plot_survival_function(cox_model, x_test, y_test, idx2test)

    #%% RSF model
    # Initialize and train the RSF model
    rsf = RandomSurvivalForest(n_estimators=100,
                            min_samples_split=10,
                            min_samples_leaf=15,
                            max_features="sqrt",
                            n_jobs=-1,
                            random_state=20)
    rsf.fit(x_train, y_train)

    # Evaluate the RSF model
    cindex_train = rsf.score(x_train, y_train)
    cindex_test = rsf.score(x_test, y_test)
    print('### RSF Model ###')
    print('cindex train:', round(cindex_train, 3))
    print('cindex test:', round(cindex_test, 3))

    # Plot survival function
    def plot_survival_function(model, x_test, y_test, idx2test):
        pred_surv = model.predict_survival_function(x_test[idx2test])
        time_points = np.arange(1, 2161)
        for i, surv_func in enumerate(pred_surv):
            plt.step(time_points, surv_func(time_points), where="post",
                    label="c={0}, ttf_comp2={1}".format(round(x_test[idx2test[i], -1]), int(y_test[idx2test[i]][1])))
        plt.ylabel("est. probability of survival $\hat{S}(t)$")
        plt.xlabel("cycles $c$")
        plt.legend(loc="best")
        plt.title('Fig.1: Predicted survival function for cycles={0}'.format(x_test[idx2test, -1]))
        plt.show()

    idx2test = [0, 10, 20]
    plot_survival_function(rsf, x_test, y_test, idx2test)


    #%% Gradient Boosting Survival Analysis
    gbsa = GradientBoostingSurvivalAnalysis(n_estimators=100, learning_rate=0.01)
    gbsa.fit(x_train, y_train)

    # Evaluate the Gradient Boosting model
    cindex_train = gbsa.score(x_train, y_train)
    cindex_test = gbsa.score(x_test, y_test)
    print('### Gradient Boosting Survival Analysis Model ###')
    print('cindex train:', round(cindex_train, 3))
    print('cindex test:', round(cindex_test, 3))

    # Define the plot_survival_function
    def plot_survival_function(model, x_test, y_test, idx2test):
        pred_surv = model.predict_survival_function(x_test[idx2test])
        time_points = np.arange(1, 2161)
        for i, surv_func in enumerate(pred_surv):
            plt.step(time_points, surv_func(time_points), where="post",
                    label="c={0}, ttf_comp2={1}".format(round(x_test[idx2test[i], -1]), int(y_test[idx2test[i]][1])))
        plt.ylabel("est. probability of survival $\hat{S}(t)$")
        plt.xlabel("cycles $c$")
        plt.legend(loc="best")
        plt.title('Fig.1: Predicted survival function for cycles={0}'.format(x_test[idx2test, -1]))
        plt.show()
    # Indices of the test samples you want to plot
    idx2test = [0, 10, 20]

    # Call the plot_survival_function
    plot_survival_function(gbsa, x_test, y_test, idx2test)


    #%% Train the SVM model
    def train_svm():
        svm_model = FastSurvivalSVM(rank_ratio=0.0, max_iter=1000, tol=1e-5, random_state=42)
        print('### FS-SVM Model ###')
        svm_model.fit(x_train, y_train)
        cindex_train = concordance_index_censored(y_train['status'], y_train['ttf_comp2'], -svm_model.predict(x_train))
        print('cindex train:', round(cindex_train[0], 3))
        cindex_test = concordance_index_censored(y_test['status'], y_test['ttf_comp2'], -svm_model.predict(x_test))
        print('cindex test:', round(cindex_test[0], 3))
        return svm_model, cindex_train, cindex_test

    svm_model, _, _ = train_svm()

    # Plot the predictions
    y_pred = svm_model.predict(x_test)
    cycle_points = x_test[:, -1]  # Use the actual cycle numbers from the test data

    # Sort the cycle points and corresponding predictions
    sorted_indices = np.argsort(cycle_points)
    cycle_points_sorted = cycle_points[sorted_indices]
    y_pred_sorted = y_pred[sorted_indices]

    plt.step(cycle_points_sorted, y_pred_sorted, where="post", label="ttf pred")
    plt.ylim([0, 1300])
    plt.xlim([0, cycle_points_sorted.max() + 1])  # Adjusted x-axis limit to max cycle number
    plt.ylabel("time until replacement $ttf$")
    plt.xlabel("cycles $c$")
    plt.legend(loc="best")
    plt.title('Predicted time until replacement for Component 2.')
    plt.show()

    # %% tests the model for each maintenance sample
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))

    for ax, idx in zip(axes.flatten(), range(len(df_minor))):
        # splits the train-test set
        x_train, y_train, x_test, y_test = dataset_split(df_sample, idx)
        # trains the model
        svm_model, cindex_train, cindex_test = train_svm()
        # plots the predictions
        y_pred = svm_model.predict(x_test)
        time_points = x_test[:, -1]
        
        # Sort the cycle points and corresponding predictions
        sorted_indices = np.argsort(time_points)
        time_points_sorted = time_points[sorted_indices]
        y_pred_sorted = y_pred[sorted_indices]
        y_test_sorted = y_test['ttf_comp2'][sorted_indices]
        
        ax.step(time_points_sorted, y_pred_sorted, where="post", label="ttf pred")
        ax.step(time_points_sorted, y_test_sorted, where="post", label="ttf real")
        ax.legend(loc="best")
        ax.set_title("{0}, cindex=({1}, {2})".format(df_minor.iloc[idx]['datetime'],
                                                    round(cindex_train[0], 3),
                                                    round(cindex_test[0], 3)))
        # Set y limits for each subplot individually
        ax.set_ylim([min(np.min(y_pred_sorted), np.min(y_test_sorted)), max(np.max(y_pred_sorted), np.max(y_test_sorted))])
        # Set x limits for each subplot individually
        ax.set_xlim([min(time_points_sorted), max(time_points_sorted)])

    fig.supylabel("time until replacement $ttf$")
    fig.supxlabel("cycles $c$")

    # Adjust the layout of subplots
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.3, hspace=0.4)

    plt.show()


    #%% plot surv function with comp of last values
    idx2compute = np.where(y_train['status'] == False)
    pred_surv = cox_model.predict_survival_function(x_train[idx2compute])
    time_points = np.arange(1, 2161)
    df_dead_surv = pd.DataFrame()

    for i, surv_func in enumerate(pred_surv):
        df_dead_surv['comp_{0}'.format(i)] = surv_func(time_points)

    df_dead_surv['mean'] = df_dead_surv.mean(axis=1)
    plt.step(time_points, df_dead_surv['mean'], where="post", label="mean replacement", linestyle='dashed')

    # Find a valid index with non-zero cycle value
    valid_index = None
    for i in range(len(x_test)):
        if x_test[i, -1] != 0:
            valid_index = i
            break

    # Ensure the valid index is found and print for debugging
    if valid_index is not None:
        print("x_test[{}] cycle value: {}".format(valid_index, x_test[valid_index, -1]))
        print("y_test[{}][1]: {}".format(valid_index, y_test[valid_index][1]))

        # Select a specific index for the test prediction
        pred_surv = cox_model.predict_survival_function(x_test[[valid_index]])
        plt.step(time_points, pred_surv[0](time_points), where="post",
                label="current pred, c={0}".format(round(x_test[valid_index, -1]), int(y_test[valid_index][1])))
    else:
        print("No valid index found with non-zero cycle value.")

    plt.ylabel("est. probability of survival $\hat{S}(t)$")
    plt.xlabel("cycles $c$")
    plt.legend(loc="best")
    plt.title('Predicted survival function for Component 2')
    plt.show()

# %%
dump(rsf, 'best_rsf_model_comp2.pkl')
dump(cox_model, 'best_coxph_model_comp2.pkl')
dump(gbsa, 'best_gbsa_model_comp2.pkl')
dump(svm_model, 'best_svm_model_comp2.pkl')

# %%
