# datasets
Datasets that I generally use for trainings, workshops



## Public data sets links
* [Facebook post data](https://insights.birdsonganalytics.com/static/demo/demobirdsong.facebook.csv) `text-mining`, `social-media`
* [Medicare hospitals data](https://data.medicare.gov/data/hospital-compare)  


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
