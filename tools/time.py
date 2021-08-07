import time
ini_time = 0.0
final_time = 0.0
total_time = '00h00m00s'
end_date_time = 'Date Hour year'
init_date_time = 'Date Hour year'


def init():
    global ini_time, init_date_time
    ini_time = time.time()
    init_date_time = time.ctime()


def end():
    global final_time, end_date_time, total_time, ini_time
    final_time = time.time()
    end_date_time = time.ctime()
    hour = 0
    minute = 0
    second = 0
    value = final_time - ini_time
    if value >= 3600:
        hour = int(value / 3600)
        helper = value % 3600
        if helper >= 60:
            minute = int(helper / 60)
            second = int(helper % 60)
        else:
            second = int(helper)
        total_time = '{0}h:{1}m:{2}s'.format(hour, minute, second)
    elif value >= 60:
        minute = int(value / 60)
        second = int(value % 60)
        total_time = '{0}h:{1}m:{2}s'.format(hour, minute, second)
    else:
        second = int(value)
        total_time = '{0}h:{1}m:{2}s'.format(hour, minute, second)


def get_execu_time():
    global total_time
    return total_time


def get_init_date_time():
    global init_date_time
    return init_date_time


def get_end_date_time():
    global end_date_time
    return end_date_time
