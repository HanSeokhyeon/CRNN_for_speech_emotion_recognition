"""

Copyright 2019- Han Seokhyeon

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""

import requests, zipfile, io, os, shutil
from loader import logger

def data_download():
    if os.path.isdir('./dataset/wav'):
        logger.info("emo-db already exist")
        return
    else:
        logger.info("emo-db downloading")
        r = requests.get('http://emodb.bilderbar.info/download/download.zip')
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall('./dataset')
        shutil.rmtree('./dataset/lablaut')
        shutil.rmtree('./dataset/labsilb')
        shutil.rmtree('./dataset/silb')
        os.remove('./dataset/erkennung.txt')
        os.remove('./dataset/erklaerung.txt')



if __name__ == '__main__':
    data_download()