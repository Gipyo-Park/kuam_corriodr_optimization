
import re

def dms_to_dd(dms_str):
    parts = re.split(r'[°\'\"NSEW]+', dms_str)
    parts = [p.strip() for p in parts if p.strip()]
    
    if len(parts) < 3:
        return None
        
    deg = float(parts[0])
    mnt = float(parts[1])
    sec = float(parts[2])
    
    return deg + mnt/60 + sec/3600

verti = ["35°36'22.07\"N", "129° 4'33.55\"E"]

waypoints_str = """
35°35'47.9961"N, 129°05'09.6189"E
35°35'40.0512"N, 129°05'52.0649"E
35°35'55.3241"N, 129°06'25.3326"E
35°36'31.0754"N, 129°06'52.6863"E
35°37'10.4089"N, 129°06'55.3529"E
35°37'18.5310"N, 129°07'35.8016"E
35°35'57.9083"N, 129°07'43.2477"E
35°35'31.2488"N, 129°07'23.9013"E
35°35'12.4715"N, 129°06'33.1775"E
35°34'45.7697"N, 129°06'11.1637"E
35°34'20.7821"N, 129°06'09.9362"E
35°33'55.2629"N, 129°05'51.0174"E
35°33'34.5251"N, 129°05'05.8784"E
35°34'01.6684"N, 129°04'39.5476"E
35°34'36.4359"N, 129°05'23.8959"E
35°34'55.2707"N, 129°05'25.5255"E
35°35'06.9137"N, 129°05'15.6041"E
35°35'17.4695"N, 129°04'52.5458"E
35°35'19.4368"N, 129°04'08.7854"E
35°35'35.4532"N, 129°03'59.6035"E
35°35'58.3259"N, 129°04'20.5049"E
35°36'10.7670"N, 129°04'21.1753"E
35°36'50.4942"N, 129°03'47.2969"E
35°37'00.7047"N, 129°03'30.1112"E
35°37'06.6663"N, 129°03'04.3955"E
35°37'29.6793"N, 129°03'13.2155"E
35°37'23.4939"N, 129°03'39.6321"E
35°37'13.6760"N, 129°03'56.5663"E
"""

v_lat = dms_to_dd(verti[0])
v_lon = dms_to_dd(verti[1])

with open("c:/Users/HMCL/Desktop/kuam_1219/kuam_1219/coords_result.txt", "w") as f:
    f.write(f"Vertiport: [{v_lat}, {v_lon}]\n")
    f.write(f"Lats: {np.array2string(np_lats, separator=', ', precision=8, suppress_small=True).replace(chr(10), '')}\n")
    f.write(f"Lons: {np.array2string(np_lons, separator=', ', precision=8, suppress_small=True).replace(chr(10), '')}\n")
