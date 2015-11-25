#!/usr/bin/env python

import sys, os
from io import BytesIO
import time

import requests
from bs4 import BeautifulSoup
from PIL import Image

def download_page(url, target_width, target_height, crop=False, pad=True, file_prefix="", target_extension="jpg", encoding="JPEG", image_links=True, image_embeds=False, delay=0.1):
	# If crop = True, we cut into a section of the image after rescaling.
	# If pad = True, we pad the image on either side.
	print("Getting URL {}".format(url))
	result = requests.get(url)
	soup = BeautifulSoup(result.content, 'html.parser')
	# Get image URL
	print("Found {} links.".format(len(soup.find_all('a'))))
	print("Found {} embeds.".format(len(soup.find_all('img'))))
	index = 0
	if image_links:
		for link in soup.find_all('a'):
			# Pull URL from hyperlink
			img_src = link.get('href')
			if img_src is not None and img_src.endswith(target_extension):
				index = get_image(img_src, target_width, target_height, crop, pad, file_prefix, encoding, index)
				time.sleep(delay)
	if image_embeds:
		for link in soup.find_all('img'):
			img_src = link.get('src')
			if img_src is not None and img_src.endswith(target_extension):
				index = get_image(img_src, target_width, target_height, crop, pad, file_prefix, encoding, index)
				time.sleep(delay)

def get_image(img_src, target_width, target_height, crop, pad, file_prefix, target_extension, index):
	print("Getting {}".format(img_src))
	# Fix the head.
	if img_src.startswith("/"):
		img_src = img_src[1:]
	if img_src.startswith("/"):
		img_src = img_src[1:] # Some sites start with "//blah"
	if not (img_src.startswith("http://") or img_src.startswith("https://")):
		img_src = "http://" + img_src
	# Download image
	try:
		result2 = requests.get(img_src)
	except MissingSchema:
		return index # Bad parse.  Retry URL?
	# Download image
	img = None
	try:
		img = Image.open(BytesIO(result2.content))
	except IOError:
		return index
	# Write image after resize.
	w = float(img.size[0])
	h = float(img.size[1])
	newimg = None
	if pad: # Pad the outside of the image.
		# Calculate new size
		max_res = max(w, h)
		new_width = int(target_width*float(w/max_res))
		new_height = int(target_height*float(h/max_res))
		# Center image in new image.
		newimg = Image.new(img.mode, (target_width, target_height))
		offset_x = int((target_width/2)-(new_width/2))
		offset_y = int((target_height/2)-(new_height/2))
		box = (offset_x, offset_y, offset_x+new_width, offset_y+new_height)
		newimg.paste(img.resize((new_width, new_height)), box)
	elif crop: # Cut a section from the middle of the image.
		# Calculate size
		res_cap = min(w, h)
		new_width = int(target_width*(w/float(res_cap)))
		new_height = int(target_height*(h/float(res_cap)))
		# Cut image chunk.
		offset_x = int((new_width/2)-(target_width/2))
		offset_y = int((new_height/2)-(target_height/2))
		newimg = img.resize(
			(new_width, new_height)
		).crop(
			(offset_x, offset_y, offset_x+target_width, offset_y+target_height)
		)
	else: # Just write it.
		newimg = img
	# Find a name.
	filename = file_prefix + str(index) + "." + target_extension
	while os.path.isfile(filename):
		index += 1
		filename = file_prefix + str(index) + "." + target_extension
	newimg.save(filename)
	print("Wrote file {}".format(filename))
	return index+1

def main():
	while True:
		url = raw_input("URL: ")
		if not url:
			return False;
		prefix = raw_input("Prefix: ")
		embedded = (raw_input("Embedded images (y/[n]): ") == 'y')
		linked = (raw_input("Linked images (y/[n]): ") == 'y')
		download_page(url, 256, 256, pad=True, encoding="JPEG", image_links=linked, image_embeds=embedded, file_prefix=prefix)

if __name__=="__main__":
	main()
