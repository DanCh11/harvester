import os
import logging

import pandas as pd

from scrapy import Request, Spider
from typing import Any, Iterable

from ..items import AldiReviewsItem


class AldiReviewsSpider(Spider):
    ALDI_REVIEW_FILENAME = '../../resources/crawler/aldi-reviews.csv'
    name = "aldi-reviews"
    url = "https://www.trustpilot.com/review/www.aldi.de?languages=all"
    
    def __init__(self, *args: Any, filename: str = ALDI_REVIEW_FILENAME, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.filepath = '../../resources/crawler/'
        self.filename = filename
        self.data = list()

        if not os.path.exists(self.filepath):
            os.makedirs(self.filepath)
            logging.info("---directory created---")
        else:
            logging.info("---directory already exists---")
        
    def start_requests(self) -> Iterable[Request]:
        yield Request(url=self.url, callback=self.parse)

    def parse(self, response):
        next_page = response.css('nav.pagination_pagination___F1qS a.pagination-link_next__SDNU4::attr(href)').get()
        review_containers = response.css('section.styles_reviewsContainer__3_GQw')

        comments = review_containers.css('div.styles_reviewContent__0Q2Tg p.typography_body-l__KUYFJ::text').getall()
        ratings = review_containers.css('div.star-rating_starRating__4rrcf img::attr(alt)').getall()
        posting_times = review_containers.css('time::text').getall()
        
        for posting_time, rating, comment in zip(posting_times, ratings, comments):

            item = AldiReviewsItem()
            item['posting_time'] = posting_time
            item['rating'] = rating
            item['comment'] = comment

            self.data.append(item)   
        
        if next_page is not None:
            yield response.follow(next_page, self.parse)
            
    def closed(self, reason):
        df = pd.DataFrame(self.data)
        df.to_csv(self.filename, index=False)
