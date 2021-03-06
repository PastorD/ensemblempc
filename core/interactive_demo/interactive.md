# Plot interactive graphs using python

## Packages analyzed

- Plotly: web-based, popular
- Matplotlib: no dependencies, not well documented, slow, ugly
- Bokeh: web-based, built on top of Matplotlib
- Pyqtgraph: fast, ugly, it can use other pyqt widgets 

Google Trends Comparison:
![comparision](comparision.png)

Plotly is by far the most popular, followed by Bokeh (I had to include the term python to all searches to avoid other meanings of Bokeh)

## Plotly

```bash
pip install plotly
```

It looks nice, easy to do simple things but harder for more complicated things [example](https://plot.ly/python/sliders/)

![plotly](plotly_slider.gif)

## Matplotlib
It does not look good and it is slow, but it is the easiest to use [example](https://matplotlib.org/gallery/widgets/slider_demo.html)

![matplotlib](matplotlib_slider.gif)

## Bokeh

```bash
pip install bokeh
```

It looks very good but it is slow, [example](https://docs.bokeh.org/en/latest/docs/gallery/slider.html)

![bokeh](bokeh_slider.gif)

## Pyqtgraph

```bash
pip install pyqt5 pyqtgraph
```

It is very fast and the syntax is easy, [example](https://stackoverflow.com/questions/42007434/slider-widget-for-pyqtgraph/42011414)

![pyqtgraph](pyqtgraph_slider.gif)

# First Prototype to our problem

Based on the previous analysis, plotly and pyqtgraph were tested for our data:

## 1D Landing Plotly Example

It is easy to plot one variable, but I struggled to plot more than one. It looks very slow even with only one variable. The slider controls the point in time to show the predictions.

![1d_plotly](1d_plotly.gif)

## 1D Landing Pyqtgraph Example

It was easy to change the plot and it is very fast. The second slider controls the point in time to show the predictions. The third slider controls the episode number.

![1d_pyqtgraph](1d_pyqtgraph.gif)

# EnMPC Analysis 0.1

After the problems with plotly, we decided to implement the first version of the analyzer in Pyqtgraph. See below a video showing its operation: blue shows the closed loop trajectory and red are the ensemble predictions. One slider controls episode number while the other slider controls the point along the trajectory to visualize the ensemble predictions. Note that for the input there is only one prediction.

![1d_pyqtgraph_2](1d_pyqtgraph_2.gif)

If we activate the ground constraint it will avoid the ground for all possible dynamics in the ensemble. Note how the ensembles predictions crosses. 

![1d_pyqtgraph_enMPC](1d_pyqtgraph_enMCP.gif)
