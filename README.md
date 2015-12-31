# The Efficient Frontier of Philanthropy

The concept of an [efficient frontier](https://en.wikipedia.org/wiki/Efficient_frontier) for investment portfolios originated with Harry Markowitz's [Modern Portfolio Theory](https://en.wikipedia.org/wiki/Modern_portfolio_theory).  Markowitz's main insight was that you can minimize the risk you take for any level of return by diversifying a portfolio.  The end result is a hyperbola, often called the Markowitz Bullet, that demonstrates that greater returns are associated with higher risk.  Although this approach to porfolio optimization is less popular today, modified versions of MPT still underly many robo-advisors on the market today ([Wealthfront](http://www.slideshare.net/wealthfront/engineering-your-portfolio-with-etfs/35-Want_us_to_do_this), [WiseBanyan](https://wisebanyan.com/investment-strategy), [Betterment](https://www.betterment.com/portfolio/)). 

![frontier image](https://commons.wikimedia.org/wiki/File:Markowitz_frontier.jpg#/media/File:Markowitz_frontier.jpg?raw=true =200x300)  

<img src="https://commons.wikimedia.org/wiki/File:Markowitz_frontier.jpg#/media/File:Markowitz_frontier.jpg"/>

The following code implements this concept in Python, and is based on a [blog post](http://blog.quantopian.com/markowitz-portfolio-optimization-2/) from Quantopian.  Their original code returns an optimal portfolio for any given level of risk when given a list of historic returns for a group of assets.  I then extend the concept of an efficient frontier to the area charitable giving, and refactor the code to take a list of charitable returns (in DALYs) and a covariance matrix as an input.  The final output is a philanthropic efficient frontier.  

See the [jupyter notebook](https://github.com/psthomas/efficient-frontier/blob/master/efficient_frontier.ipynb) for further explanation, and visit this project on the [Jupyter nbviewer](https://nbviewer.jupyter.org/github/psthomas/efficient-frontier/blob/master/efficient_frontier.ipynb) to see the interactive plots.  This code is a rough sketch of my concept, feel free to critique and contribute!    
