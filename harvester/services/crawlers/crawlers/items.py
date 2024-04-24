# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class AldiReviewsItem(scrapy.Item):
    posting_time = scrapy.Field()
    rating = scrapy.Field()
    comment = scrapy.Field()
