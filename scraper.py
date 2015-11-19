#!/usr/bin/env python

import sys, os
from io import BytesIO

import requests
from bs4 import BeautifulSoup
from PIL import Image

def download_page(url, target_width, target_height, crop=False, embed=True):
	# If crop = True, we cut into a section of the image after rescaling.
	# If embed = True, we pad the image on either side.
	if not crop ^ embed:
		print("WARNING: Need crop or rescale.")
	result = requests.get(url)
	soup = BeautifulSoup(result.content, 'html.parser')
	# Get image URL
	for link in soup.find_all('a'):
		# Pull URL from hyperlink
		if link.get('href').endswith(".png"):
			img_src = link.get('href')
		else:
			continue
		# Fix the head.
		if img_src.startswith("/"):
			img_src = img_src[1:]
		if img_src.startswith("/"):
			img_src = img_src[1:] # Some sites start with "//blah"
		if not (img_src.startswith("http://") or img_src.startswith("https://"):
			img_src = "http://" + img_src
		# Download image
		try:
			result2 = requests.get(img_src)
		except MissingSchema:
			continue # Bad parse.  Retry URL?
		# Download image
		img = None
		try:
			img = Image.open(BytesIO(result2.content))
		except IOError:
			continue
		# Write image after resize.
		w = float(img.size[0])
		h = float(img.size[1])
		newimg = None
		if embed: # Pad the outside of the image.
			# Calculate new size
			max_res = max(w, h)
			new_width = target_width*float(w/max_res)
			new_height = target_height*float(h/max_res)
			# Center image in new image.
			newimg = Image.new(img.mode, (target_width, target_height))
			offset_x = (target_width/2)-(new_width/2)
			offset_y = (target_height/2)-(new_height/2)
			box = (offset_x, offset_y, offset_x+new_width, offset_y+new_height)
			newimg.paste(img.resize((new_width, new_height))
		elif crop: # Cut a section from the middle of the image.
			# Calculate size
			res_cap = min(w, h)
			new_width = target_width*(w/float(res_cap))
			new_height = target_height*(h/float(res_cap))
			# Cut image chunk.
			offset_x = (new_width/2)-(target_width/2)
			offset_y = (new_height/2)-(target_height/2)
			newimg = img.resize(
				(new_width, new_height)
			).crop(
				(offset_x, offset_y, offset_x+target_width, offset_y+target_height)
			)
		else: # Just write it.
			newimg = img
