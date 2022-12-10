# import packages

from TikTokApi import TikTokApi as tiktok
import json 


# seting instance 
verifyFp = "nM0QbcHopd91oBwWCA2mPJ0PN7k3E5tU9ELmS-ExEmfxcG9DrKRdxUZjtjOjzqwfHCqmNI13Jir-LUZo29aYXg-2zJIvb33lqx53LA_vRSNUk5Xp-kvJFGk8Vp4doSmoSmz0qw=="
api = tiktok(custom_verifyFp=verifyFp,use_test_endpoints=True)

# get information 
trending = api.by_hashtag('Python')
print(trending)