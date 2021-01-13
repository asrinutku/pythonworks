import random
import time
start_time = time.time()

def merge_sort(mainarray):
    size = len(mainarray)
    random.seed(532)
    if size > 1:

        # mainarrayın orta index sayisini bulup main arrayi sag ve sol olarak ikiye boluyoruz
        orta = size // 2

        # orta index bulunduktan sonra diziyi sag ve sol olmak uzere 2 ye ayırıyoruz
        solarray = mainarray[:orta]
        sagarray = mainarray[orta:]

        # DİZİYİ PARCALAMA İSLEMİ

        # recursive ozellık bize diziyi 1 den fazla kez sag ve sol olarak parcalamamızı saglayacak

        # sol tarafta kalan arrayi tekrar merge sort fonk. gonderıyoruz.
        merge_sort(solarray)

        # sag tarafta kalan arrayi tekrar merge sort fonk. gonderıyoruz.
        merge_sort(sagarray)



        # sol dizi , sag dizi , maindizi counterları
        p = 0 #sol
        q = 0 #sag
        r = 0 #main

        # sag ve sola parcalanan dizilerin buyukluklerını aldık
        solsize = len(solarray)
        sagsize = len(sagarray)

        # p sol , q da sag dizi icin counter
        # amacımız sag ve sol dizilerdeki elemanları karşılaştırarak gerekli index degisimlerini yapmak
        # sag veya sol dizide counterlardan biri dizinin eleman sayisina ulasana kadar bu degisimleri yapacagız
        while p < solsize and q < sagsize:

            if solarray[p] < sagarray[q]:

                # sol taraftaki dizinin ilk elemanı sag dizinin ilk elemanından kucukse
                # mainarrayin ilk elemanı olarak tanımlıyoruz
                mainarray[r] = solarray[p]
                # sol arrayin sonraki indexine gecmek icin p yi 1 arttırıyoruz
                p += 1

            else:

                # sol taraftaki dizinin ilk elemanı sag dizinin ilk elemanından buyukse
                # mainarrayin ilk elemanı olarak sag arrayin ilk elemanı olarak atariz
                mainarray[r] = sagarray[q]
                # sag arrayin index kontrolu icin q yu 1 arttırıyoruz
                q += 1

            # main arrayde if-else yapısı ıcınde kesin bir degisiklik olacagı icin
            # main array counterını 1 arttırıyoruz
            r += 1

        #  dizi counterlarından dizinin eleman sayısına ulaşamayan taraf için boşta kalan elemanlar olacaktır
        #  üstteki while yapısında sıralamaları yaptıktan sonra
        #  boşta kalan elemanların kontrolunu yapacagız
        #  eger sol dizide boşta kalan bir eleman yoksa bu while döngüsüne girilmez
        while p < solsize:
            mainarray[r] = solarray[p]
            p += 1
            r += 1

        #  eger sag dizide boşta kalan bir eleman yoksa bu while döngüsüne girilmez
        while q < sagsize:
            mainarray[r] = sagarray[q]
            q += 1
            r += 1


mainarray = []
for i in range(0, 1000):

    mainarray.append(random.randint(0, 1000))

merge_sort(mainarray)
timeforaveragecase = (time.time() - start_time)
print(mainarray)
print("1000 elemanlı dizi AVERAGE CASE calısma suresi : ", timeforaveragecase)
print("------------")

sortedlist = sorted(mainarray)
merge_sort(sortedlist)
timeforbestcase = (time.time() - start_time)-timeforaveragecase
print(sortedlist)
print("1000 elemanlı dizi BEST CASE calısma suresi :", timeforbestcase)
print("------------")

reversesortedlist = sortedlist[::-1]
merge_sort(reversesortedlist)
timeforworstcase = (time.time() - start_time)-timeforbestcase
print(reversesortedlist)
print("1000 elemanlı dizi WORST CASE calısma suresi :", timeforworstcase)
