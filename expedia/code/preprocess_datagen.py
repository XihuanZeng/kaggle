import math
import os
import json
import pickle

os.chdir('/home/xihuan/Downloads/expedia/code')
with open(os.path.join('../data', 'dict', 'best_hotels_search_dest.pickle'), 'rb') as f:
    best_hotels_search_dest = pickle.load(f)
    best_hotels_search_dest_keys = best_hotels_search_dest.keys()
f.close()

with open(os.path.join('../data', 'dict', 'best_hotels_country.pickle'), 'rb') as f:
    best_hotels_country = pickle.load(f)
    best_hotels_country_keys = best_hotels_country.keys()
f.close()


f = open("../data/train_booking.csv", "r")
f.readline()

g1 = open('../data/train_s2.csv', 'w')
g2 = open('../data/train_s3.csv', 'w')


g1.write('date_time,user_location_country,user_location_region,user_location_city,is_mobile,is_package,srch_ci,srch_co,srch_adults_cnt,srch_children_cnt,srch_rm_cnt,srch_destination_id,hotel_country,hotel_market,hotel_cluster' + '\n')
g2.write('date_time,user_location_country,user_location_region,user_location_city,is_mobile,is_package,srch_ci,srch_co,srch_adults_cnt,srch_children_cnt,srch_rm_cnt,srch_destination_id,hotel_country,hotel_market,hotel_cluster' + '\n')

total = 0
while 1:
    line = f.readline().strip()
    total += 1

    print('Read {} lines...'.format(total))

    if line == '':
        break

    arr = line.split(",")

    if arr[0] == "" or arr[11] == "" or arr[12] == "":
        continue

    if arr[0] != "":
        book_year = int(arr[0][:4])
        book_month = int(arr[0][5:7])
        book_day = int(arr[0][8:10])
        if book_month<1 or book_month>12 or book_year<2012 or book_year>2015:
            continue


    if arr[11] != '':
        book_year = int(arr[11][:4])
        book_month = int(arr[11][5:7])
        if book_month<1 or book_month>12 or book_year<2012 or book_year>2015:
            continue

    if arr[12] != '':
        book_year = int(arr[12][:4])
        book_month = int(arr[12][5:7])
        if book_month<1 or book_month>12 or book_year<2012 or book_year>2015:
            continue
            

    date_time = arr[0]
    user_location_country = arr[3]
    user_location_region = arr[6]
    user_location_city = arr[5]
    is_mobile = arr[8]
    is_package = arr[9]
    srch_ci = arr[11]
    srch_co = arr[12]
    srch_adults_cnt = arr[13]
    srch_children_cnt = arr[14]
    srch_rm_cnt = arr[15]
    srch_destination_id = arr[16]
    hotel_country = arr[21]
    hotel_market = arr[22]
    hotel_cluster = arr[23]

    s2 = (srch_destination_id,hotel_country,hotel_market,is_package)
    s3 = (hotel_market)

    if s2 in best_hotels_search_dest_keys:
        g1.write(','.join([date_time, user_location_country, user_location_region, user_location_city, is_mobile, is_package,
                 srch_ci, srch_co, srch_adults_cnt, srch_children_cnt, srch_rm_cnt, srch_destination_id,
                 hotel_country, hotel_market, hotel_cluster]) + '\n')

    if s3 in best_hotels_country_keys:
        g2.write(','.join([date_time, user_location_country, user_location_region, user_location_city, is_mobile, is_package,
                 srch_ci, srch_co, srch_adults_cnt, srch_children_cnt, srch_rm_cnt, srch_destination_id,
                 hotel_country, hotel_market, hotel_cluster]) + '\n')

g1.close()
g2.close()
f.close()



a = '2014-08-11 07:46:59,2,3,66,348,48862,2234.2641,12,0,1,9,2014-08-27,2014-08-31,2,0,1,8250,1,0,3,2,50,628,1'
arr = a.split(',')
arr[0][8:10]
