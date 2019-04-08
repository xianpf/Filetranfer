#!/usr/bin/python
# -*- coding:utf-8 -*-

from tornado import web
from tornado import httpserver
from tornado import ioloop
import os, tempfile

haze_path = 'received_haze_images'
# 逻辑模块

class DehazeHandler(web.RequestHandler):
    def post(self, *args, **kwargs):
        received_metas = self.request.files.get("img")
        for meta in received_metas:
            filename = meta['filename']
            file_path = os.path.join(haze_path, filename)

            with tempfile.NamedTemporaryFile(prefix='haze', 
                    dir='received_haze_images', delete=False, 
                    suffix='.'+meta['content_type'].split('/')[-1]
                    ) as tfile:
                tfile.write(meta['body'])
                temp_name = tfile.name

                print(temp_name)

                
        # print(received_metas, received_metas[0].keys())
        self.write('OKxpf')

# url路由
application = web.Application([
        (r"/dehaze", DehazeHandler),
    ])

if __name__ == '__main__':
    http_server = httpserver.HTTPServer(application)
    http_server.listen(8000)
    ioloop.IOLoop.current().start()
