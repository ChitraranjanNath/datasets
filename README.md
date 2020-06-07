# datasets
Datasets that I generally use for trainings, workshops



## Public data sets links
* [Facebook post data](https://insights.birdsonganalytics.com/static/demo/demobirdsong.facebook.csv) `text-mining`, `social-media`
* [Medicare hospitals data](https://data.medicare.gov/data/hospital-compare)  
* [Uber trips data](https://github.com/fivethirtyeight/uber-tlc-foil-response)
* [Food products data](https://world.openfoodfacts.org/data)


## Code to plot decision tree
```
def draw_tree(model, columns):
    import pydotplus
    from sklearn.externals.six import StringIO
    from IPython.display import Image
    import os
    from sklearn import tree
    
    graphviz_path = 'C:\Program Files (x86)\Graphviz2.38/bin/'
    os.environ["PATH"] += os.pathsep + graphviz_path

    dot_data = StringIO()
    tree.export_graphviz(model,
                         out_file=dot_data,
                         feature_names=columns)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    return Image(graph.create_png())
```
## Code to calculate Root Mean Square Percentage Error (RMSPE)
```
# Credit: kaggle.com
def ToWeight(y):
    w = np.zeros(y.shape, dtype=float)
    ind = y != 0
    w[ind] = 1./(y[ind]**2)
    return w

def rmspe(y, yhat):
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean( w * (y - yhat)**2 ))
    return rmspe
```
