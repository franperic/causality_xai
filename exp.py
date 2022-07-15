import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportions_ztest

from sklearn.metrics import auc
from sklearn.model_selection import train_test_split
import xgboost as xgb

from sklearn.tree import DecisionTreeClassifier, plot_tree
from lightgbm import LGBMRegressor


import causalml
from causalml.optimize import PolicyLearner
from causalml.inference.meta import BaseXRegressor
from causalml.inference.tree import UpliftRandomForestClassifier, UpliftTreeClassifier
from causalml.metrics import plot_gain

df = pd.read_csv("data/uplift-v2.1.csv")
df.head()

df.shape
df.treatment.value_counts()
df.treatment.value_counts(normalize = True)

df.groupby("treatment").conversion.mean()

fig, axs = plt.subplots(12, 1, sharex=True)
for i in np.arange(12):
    feature = df.columns[i]
    axs[i].hist(df[feature])
plt.show()

corr = df.corr().to_numpy()
fig, ax = plt.subplots()
im = ax.imshow(corr)
ax.set_xticks(np.arange(len(df.columns)), labels=df.columns)
ax.set_yticks(np.arange(len(df.columns)), labels=df.columns)
# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
# Loop over data dimensions and create text annotations.
for i in np.arange(len(df.columns)):
    for j in np.arange(len(df.columns)):
        text = ax.text(j, i, round(corr[i, j], 1),
                       ha="center", va="center", color="w")
leg = plt.legend()
plt.show()

ate = df.groupby("treatment").conversion.agg(avg_conv=("mean"), sum_conv=("sum"))
plt.bar(x=ate.index, height=ate.avg_conv, tick_label=ate.index)
plt.title("ATE")
plt.show()

click = df.groupby("treatment").visit.agg(avg_click=("mean"))
plt.bar(x=click.index, height=click.avg_click, tick_label=click.index)
plt.title("Clickthrough")
plt.show()

proportions_ztest(count=ate["sum_conv"], nobs=df.treatment.value_counts())

policy_learner = PolicyLearner(policy_learner=DecisionTreeClassifier(max_depth=2), calibration=True)
X = df.iloc[:, 0:12]
W = df.treatment
Y = df.visit
policy_learner.fit(X, W, Y)




train, test  = train_test_split(df, test_size=0.2, random_state=42, stratify=df['treatment'])



# Random Undersampling (finding the majority class and undersampling it)
def random_under(df:pd.DataFrame, feature):
    
    target = df[feature].value_counts()
    
    if target.values[0]<target.values[1]:
        under = target.index.values[1]
    
    else: 
        under = target.index.values[0]
        
    df_0 = df[df[feature] != under]
    df_1 = df[df[feature] == under]
    
    df_treatment_under = df_1.sample(len(df_0))
    df_1 = pd.concat([df_treatment_under, df_0], axis=0)
    
    return df_1



train = random_under(train, 'treatment')



fig = plt.figure(figsize = (10,6))
new_target_count = train['treatment'].value_counts()
print('Class 0:', new_target_count[0])
print('Class 1:', new_target_count[1])
print('Proportion:', int(round(new_target_count[0] / new_target_count[1])), ': 1')
new_target_count.plot(kind='bar', title='Target Class Distribution', color=['#2077B4', '#FF7F0E'], fontsize = 15)
plt.xticks(rotation=0) 
plt.show()


# Function to declare Target Class

def target_class(df, treatment, target):
    
    #CN:
    df['target_class'] = 0 
    #CR:
    df.loc[(df[treatment] == 0) & (df[target] != 0),'target_class'] = 1 
    #TN:
    df.loc[(df[treatment] != 0) & (df[target] == 0),'target_class'] = 2 
    #TR:
    df.loc[(df[treatment] != 0) & (df[target] != 0),'target_class'] = 3 
    return df



train = target_class(train.drop(columns = ['conversion', 'exposure']), 'treatment', 'visit')
test = target_class(test.drop(columns = ['conversion', 'exposure']), 'treatment', 'visit')



X_train = train.drop(['visit','target_class'],axis=1)
y_train = train['target_class']
X_test = test.drop(['visit','target_class'],axis=1)
y_test = test['target_class']



def uplift_model(X_train,
                 X_test,
                 y_train,
                 y_test,
                 treatment_feature):

    result = pd.DataFrame(X_test).copy()    
    uplift_model = xgb.XGBClassifier().fit(X_train.drop(treatment_feature, axis=1), y_train)
    
    uplift_proba = uplift_model.predict_proba(X_test.drop(treatment_feature, axis=1))
    
    result['p_cn'] = uplift_proba[:,0] 
    result['p_cr'] = uplift_proba[:,1] 
    result['p_tn'] = uplift_proba[:,2] 
    result['p_tr'] = uplift_proba[:,3]
    
    result['uplift_score'] = result.eval('\
    p_cn/(p_cn + p_cr) \
    + p_tr/(p_tn + p_tr) \
    - p_tn/(p_tn + p_tr) \
    - p_cr/(p_cn + p_cr)')  

    # Put the result 
    result['target_class'] = y_test
    
    return result

result = uplift_model(X_train, X_test, y_train, y_test, 'treatment')
result.head()


plt.figure(figsize = (10,6))
plt.xlim(-.05, .1)
plt.hist(result.uplift_score, bins=1000, color=['#2077B4'])
plt.xlabel('Uplift score')
plt.ylabel('Number of observations in validation set')
plt.show()

def qini_rank(uplift): 
    # Function to Rank the data by the uplift score
    ranked = pd.DataFrame({'ranked uplift':[], 'target_class':[]})
    ranked['target_class'] = uplift['target_class']
    ranked['uplift_score'] = uplift['uplift_score']
    ranked['ranked uplift'] = ranked.uplift_score.rank(pct=True, ascending=False)
    # Data Ranking   
    ranked = ranked.sort_values(by='ranked uplift').reset_index(drop=True)
    return ranked

def qini_eval(ranked):
    uplift_model, random_model = ranked.copy(), ranked.copy()
    # Using Treatment and Control Group to calculate the uplift (Incremental gain)
    C, T = sum(ranked['target_class'] <= 1), sum(ranked['target_class'] >= 2)
    ranked['cr'] = 0
    ranked['tr'] = 0
    ranked.loc[ranked.target_class == 1,'cr'] = 1
    ranked.loc[ranked.target_class == 3,'tr'] = 1
    ranked['cr/c'] = ranked.cr.cumsum() / C
    ranked['tr/t'] = ranked.tr.cumsum() / T
    # Calculate and put the uplift and random value into dataframe
    uplift_model['uplift'] = round(ranked['tr/t'] - ranked['cr/c'],5)
    random_model['uplift'] = round(ranked['ranked uplift'] * uplift_model['uplift'].iloc[-1],5)
    
    uplift_model['Number_of_exposed_customers'] = np.arange(len(uplift_model))+1
    uplift_model['visits_gained'] = uplift_model.uplift*len(uplift_model)
    
    # Add q0
    q0 = pd.DataFrame({'ranked uplift':0, 'uplift':0, 'target_class': None}, index =[0])
    uplift_model = pd.concat([q0, uplift_model]).reset_index(drop = True)
    random_model = pd.concat([q0, random_model]).reset_index(drop = True)  
    # Add model name & concat
    uplift_model['model'] = 'Uplift model'
    random_model['model'] = 'Random model'
    merged = pd.concat([uplift_model, random_model]).sort_values(by='ranked uplift').reset_index(drop = True)
    return merged, uplift_model

def uplift_curve(uplift_model):
    plt.figure(figsize = (10,6))
    # plot the data
    ax = uplift_model['visits_gained'].plot(color=['#2077B4'])
    # Plot settings
    #sns.set_style('whitegrid')
    handles, labels = ax.get_legend_handles_labels()
    plt.xlabel('Number of customers treated')
    plt.ylabel('Incremental visits')
    plt.grid(b=True, which='major')
    return ax

def qini_plot(merged:pd.DataFrame, uplift_model:pd.DataFrame):
    gain_x = uplift_model['ranked uplift']
    gain_y = uplift_model.uplift
    qini = auc(gain_x, gain_y)
    # plot the data
    plt.figure(figsize = (10,6))
    #mpl.rcParams['font.size'] = 8
    qini = auc(gain_x, gain_y)

    ax = plt.plot(gain_x, gain_y, color= '#2077B4',
        label='Normalized Uplift Model, Qini Score: {}'.format(round(qini,2)))
    
    plt.plot([0, gain_x.max()], [0, gain_y.max()],
        '--', color='tab:orange',
        label='Random Treatment')
    plt.legend()
    plt.xlabel('Porportion Targeted')
    plt.ylabel('Uplift')
    plt.grid(b=True, which='major')

    return ax

def plot_uplift(result:pd.DataFrame):
    # Function to plot the uplift curve
    ranked = qini_rank(result)
    merged, uplift_model = qini_eval(ranked)
    ax1 = uplift_curve(uplift_model)
    
    return ax1

def plot_qini(result:pd.DataFrame):
    # Function to plot the qini curve
    ranked = qini_rank(result)
    merged, uplift_model = qini_eval(ranked)
    ax2 = qini_plot(merged, uplift_model)
    
    return ax2 




plot_uplift(result)
plot_qini(result)


df.pivot_table(
    values="conversion",
    index="treatment",
    aggfunc=[np.mean, np.size],
    margins=True
)

model_df = df.copy()
model_df["is_treated"] = model_df.treatment
model_df.treatment = np.where(model_df.treatment==1, "treatment", "control")
model_df.drop(columns=["visit", "exposure"], inplace=True)
df_train, df_test = train_test_split(model_df, test_size=0.2, random_state=42)





clf = UpliftTreeClassifier(control_name='control')
clf.fit(df_train.iloc[:, :12].values,
         treatment=df_train['treatment'].values,
         y=df_train['conversion'].values)
p = clf.predict(df_test.iloc[:, :12].values)

df_res = pd.DataFrame(p, columns=clf.classes_)
df_res.head()


uplift_model = UpliftRandomForestClassifier(control_name='control')

uplift_model.fit(df_train.iloc[:, :12].values,
                 treatment=df_train['treatment'].values,
                 y=df_train['conversion'].values)

df_res = uplift_model.predict(df_test.iloc[:, :12].values, full_output=True)
print(df_res.shape)
df_res.head()



y_pred = uplift_model.predict(df_test.iloc[:, :12].values)

y_pred.shape

result = pd.DataFrame(y_pred,
                      columns=uplift_model.classes_[1:])
result.head()

pred_overview = pd.DataFrame({
    "conversion": df_test.conversion,
    "is_treated": df_test.is_treated,
    "uplift_model": df_res.delta_treatment
})

plot_gain(pred_overview, outcome_col="conversion", treatment_col="is_treated")
plt.show()
