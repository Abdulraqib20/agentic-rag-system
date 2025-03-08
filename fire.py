### This script demonstrates how to use the Firecrawl API to generate long-form content from a URL. ###
### The Firecrawl API is a web scraping tool that can extract content from web pages and generate long-form content. ###
### The API can be used to generate blog posts, articles, and other types of content. ###
### In this example, we will use the Firecrawl API to generate long-form content from a blog URL. ###
### We will specify the maximum number of URLs to crawl and whether to show the full text. ###


from firecrawl import FirecrawlApp

from config.appconfig import FIRECRAWL_API_KEY

firecrawl = FirecrawlApp(api_key=FIRECRAWL_API_KEY)
params = {
    "maxUrls": 2,
    "showFullText": True
}
results = firecrawl.generate_llms_text(
    url="https://www.blog.dailydoseofds.com",
    params=params
)
print(f"Generated Data: {results['data']}")
