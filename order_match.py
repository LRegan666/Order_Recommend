import datetime

import numpy as np
import pandas as pd
from xgboost.sklearn import XGBClassifier
from sqlalchemy import create_engine

from utils import sex_map, subject_map, term_map, tidbReader, consumer
from match import time_to_seconds, get_conditions
from features import get_features, process_features

np.random.seed(0)


def book_version_encode(book_version):
    book_version_convert = []
    book_version = book_version.strip('[]').replace('"', '').split(',')
    for book in book_version:
        book = book.encode('utf8').decode('utf8')
        book_version_convert.append(book)
    book_version = ' '.join(book_version_convert)
    return book_version


def conditions_optimization(conditions):
    conditions['sex'] = sex_map[conditions['sex']]
    conditions['subject'] = subject_map[conditions['subject']]
    if '初' in conditions['grade']:
        conditions['grade'] = term_map['初中']
    elif '高' in conditions['grade']:
        conditions['grade'] = term_map['高中']
    else:
        conditions['grade'] = term_map['小学']
    return conditions


def order_scheduler(conditions, recievers):
    """订单分流策略，A\B订单固定概率分流，C订单根据全\专\基地老师的任务完成情况随机分流"""
    conditions = conditions_optimization(conditions)
    if conditions['order_level'] == 'A':
        recievers[0]['orders'].append(conditions)
    elif conditions['order_level'] == 'B':
        direction = direct_generator([0.3, 0.7], ['A', 'B'])
        if direction == 'A':
            recievers[0]['orders'].append(conditions)
        else:
            recievers[1]['orders'].append(conditions)
    elif conditions['order_level'] == 'C':
        direction = c_flow_guider(recievers[0], recievers[1], recievers[2])
        if direction == 'A':
            recievers[0]['orders'].append(conditions)
        elif direction == 'B':
            recievers[1]['orders'].append(conditions)
        else:
            recievers[2]['orders'].append(conditions)
    else:
        recievers[3]['orders'].append(conditions)
    return recievers


def teachers_allocator(recievers, units, units_spare, conn, start_status, month_start=None):
    """创建并维护全/专/基/平老师信息队列，根据订单选择符合条件老师"""
    if start_status:
        units, units_spare = teachers_unit_init(conn)
    teacher_list, units, units_spare = order_consumer(recievers, units, units_spare, conn)
    units, units_spare = teachers_unit_update(units, units_spare)
    if inspect_timer(month_start, mode='week'):
        recievers = teacher_unit_status_check(recievers, units, month_start)
    if inspect_timer(None, mode='day'):
        new_teachers = teachers_increment_collector(conn)
        if new_teachers.shape[0] > 0:
            units = team_add_allocator(units, new_teachers)
    teachers = teacher_drop_duplicate(teacher_list)
    return teachers, recievers, units, units_spare


def team_add_allocator(units, new_teachers):
    """检测新增老师并添加到相应老师信息队列"""
    new_teachers['bookVersion'] = new_teachers.bookVersion.apply(book_version_encode)
    new_teachers_full = new_teachers[new_teachers.job_type == 2]
    new_teachers_profession = new_teachers[new_teachers.job_type == 3]
    new_teachers_base = new_teachers[new_teachers.job_type.isin([4, 5, 6])]
    new_teachers_part_time = new_teachers[new_teachers.job_type == 1]
    new_units = [new_teachers_full, new_teachers_profession,
                     new_teachers_base, new_teachers_part_time]
    uind = 0
    while uind < len(new_units):
        if new_units[uind].shape[0] > 0:
            new_units[uind] = teacher_rank_replace(new_units[uind])
            new_units[uind] = new_units[uind].drop_duplicates()
            new_units[uind] = new_units[uind].sort_values(by=['rate_type'])
            new_units[uind] = unit_inner_sort(new_units[uind])
            new_units[uind]['order_nums'] = 0
            units[uind] = pd.concat([units[uind], new_units[uind]], ignore_index=True)
        uind += 1
    return units


def teacher_drop_duplicate(teacher_list):
    teachers = []
    for teacher in teacher_list:
        if teacher not in teachers:
            teachers.append(teacher)
        if len(teachers) == 3:
            break
    return teachers


def teachers_increment_collector(conn):
    now = datetime.datetime.now()
    date = now.strftime('%Y-%m-%d')
    increment_sql = """
    select t1.tid, t2.sex, t2.bookVersion, t2.rate_type, t2.job_type,
    t2.teacher_type, substr(t2.identity_no,7,4) as birth_year,
    t2.school_age, t1.term, t1.subject from fudao_account.fudao_teacher as t1
    inner join fudao_account.fudao_teacher_info as t2 on t1.tid=t2.tid
    where t1.sign_status=1 and t1.create_time like '{}%'
    """.format(date)
    teachers_df = pd.read_sql(increment_sql, conn)
    return teachers_df


def teacher_unit_status_check(recievers, units, month_start):
    """老师当前任务完成情况检查"""
    now = datetime.datetime.now()
    start = month_start
    interval_days = (now-start).days
    for uind in range(4):
        if uind == 0:
            primary = round(units[uind][units[uind].term == 1].order_nums.mean() / interval_days, 1)
            junior = round(units[uind][units[uind].term == 7].order_nums.mean() / interval_days, 1)
            high = round(units[uind][units[uind].term == 10].order_nums.mean() / interval_days, 1)
            if primary < 4.5 or junior < 4.7 or high < 5:
                recievers[uind]['status'] = 0
            else:
                recievers[uind]['status'] = 1
        elif uind == 1 or uind == 2:
            teacher_order = round(units[uind].order_nums.mean() / interval_days, 1)
            if teacher_order < 1.3:
                recievers[uind]['status'] = 0
            else:
                recievers[uind]['status'] = 1
        else:
            continue
    return recievers


def inspect_timer(start_time, mode=None):
    check_state = False
    start = None
    if start_time is not None:
        start = start_time
    current = datetime.datetime.now()
    if current.hour == 23 and current.minute < 15:
        if mode == 'week':
            interval_days = (current-start).days
            if interval_days > 0 and interval_days % 7 == 0:
                check_state = True
        if mode == 'day':
            check_state = True
    return check_state


def teachers_unit_update(units, units_spare):
    """过滤信息队列中任务完成的老师"""
    for uind in range(4):
        tind_list = []
        if units[uind].shape[0] > 0:
            if uind == 0:
                for ind in range(units[uind].shape[0]):
                    if units[uind].iloc[ind]['term'] == 1 and units[uind].iloc[ind]['order_nums'] >= 130:
                        units_spare[uind], tind_list = teacher_transfer(units[uind], units_spare[uind], tind_list, ind)
                    if units[uind].iloc[ind]['term'] == 7 and units[uind].iloc[ind]['order_nums'] >= 140:
                        units_spare[uind], tind_list = teacher_transfer(units[uind], units_spare[uind], tind_list, ind)
                    if units[uind].iloc[ind]['term'] == 10 and units[uind].iloc[ind]['order_nums'] >= 150:
                        units_spare[uind], tind_list = teacher_transfer(units[uind], units_spare[uind], tind_list, ind)
                for tind in tind_list:
                    units[uind] = units[uind].drop(tind)
                units[uind] = units[uind].reset_index(drop=True)
            elif uind == 1 or uind == 2:
                for ind in range(units[uind].shape[0]):
                    if units[uind].iloc[ind]['order_nums'] >= 40:
                        units_spare[uind], tind_list = teacher_transfer(units[uind], units_spare[uind], tind_list, ind)
                for tind in tind_list:
                    units[uind] = units[uind].drop(tind)
                units[uind] = units[uind].reset_index(drop=True)
            else:
                continue
    return units, units_spare


def teacher_transfer(units, units_spare, tind_list, ind):
    units_spare.append(units.iloc[ind], ignore_index=True)
    tind = units.iloc[ind].index.tolist()[0]
    tind_list.append(tind)
    return units_spare, tind_list


def order_consumer(recievers, units, units_spare, conn):
    """消费订单"""
    teacher_ids = []
    for ind in range(4):
        if recievers[ind]['orders']:
            order_info = recievers[ind]['orders'].pop()
            teachers_idle_time, teachers_course_time = get_idle_time(order_info['classtime'], order_info['week'], conn)
            teacher_ids, units[ind], units_spare[ind] = teacher_selector(order_info, teachers_idle_time,
                                           teachers_course_time, units[ind], units_spare[ind], teacher_ids)
            if len(teacher_ids) < 3:
                if ind < 3:
                    recievers[ind+1]['orders'].append(order_info)
                else:
                    break
            if len(teacher_ids) == 3:
                break
    return teacher_ids, units, units_spare


def get_idle_time(specified_time, specified_week, conn):
    idle_time_sql = """
    select tid, start_time, end_time 
    from fudao_course.fudao_teacher_idletime
    where week={}
    """.format(specified_week)
    course_time_sql = """
    select tid, day_start_time, day_end_time 
    from fudao_course.fudao_course
    where day='{}'
    """.format(specified_time[:10])
    idle_time = pd.read_sql(idle_time_sql, conn)
    course_time = pd.read_sql(course_time_sql, conn)
    return idle_time, course_time


def teacher_selector(order_info, idle_time, course_time, unit, unit_spare, teacher_list):
    """选择合适老师"""
    no_specified = True
    teacher_list, unit, specified_status = query_operator(order_info, unit, idle_time, course_time, teacher_list,
                                                          method='strict', is_spare=False, is_specified=no_specified)
    if no_specified:
        if len(teacher_list) < 3:
            teacher_list, unit, specified_status = query_operator(order_info, unit, idle_time, course_time, teacher_list,
                                                                  method='loose', is_spare=False, is_specified=no_specified)
        if len(teacher_list) < 3:
            teacher_list, unit_spare, specified_status = query_operator(order_info, unit_spare, idle_time, course_time, teacher_list,
                                                                        method='strict', is_spare=True, is_specified=no_specified)
        if len(teacher_list) < 3:
            teacher_list, unit_spare, specified_status = query_operator(order_info, unit_spare, idle_time, course_time, teacher_list,
                                                                        method='loose', is_spare=True, is_specified=no_specified)
    if teacher_list:
        if unit[unit.tid == teacher_list[0]].shape[0] > 0:
            ind = unit[unit.tid == teacher_list[0]].index.tolist()[0]
            unit.loc[ind, 'order_nums'] += 1
        if unit_spare[unit_spare.tid == teacher_list[0]].shape[0] > 0:
            ind = unit_spare[unit_spare.tid == teacher_list[0]].index.tolist()[0]
            unit_spare.loc[ind, 'order_nums'] += 1
    return teacher_list, unit, unit_spare


def query_operator(order_info, unit, idle_time, course_time,
                   teacher_list, method=None, is_spare=False, is_specified=True):
    if is_specified and order_info['specified_tids']:
        is_specified = False
        for teacher_info in order_info['specified_tids']:
            teacher_list.append(teacher_info['tid'])
            if len(teacher_list) == 3:
                break
        return teacher_list, unit, is_specified
    for ind in range(unit.shape[0]):
        state = condition_compare(order_info, unit.iloc[ind], idle_time, course_time, mode=method)
        if state and unit.iloc[ind]['tid'] not in teacher_list:
            teacher_list.append(unit.iloc[ind]['tid'])
        if len(teacher_list) == 3:
            break
    if not is_spare:
        # for tid in teacher_list:
        if teacher_list:
            tid = teacher_list[0]
            if unit[unit.tid == tid].shape[0] > 0:
                cond, ind = unit[unit.tid == tid], unit[unit.tid == tid].index.tolist()[0]
                unit = unit.drop(ind)
                unit = unit.append(cond, ignore_index=True)
    return teacher_list, unit, is_specified


def condition_compare(order_info, teacher_info, idle_time, course_time, mode=None):
    """订单条件匹配"""
    state = False
    if order_info['grade'] == teacher_info['term'] \
            and order_info['subject'] == teacher_info['subject'] \
            and order_info['book_version'] in teacher_info['bookVersion']:
        if order_info['sex'] == 2:
            if mode == 'strict':
                if (order_info['teacher_category'] == '一线老师' and teacher_info['teacher_type'] in (0, 1)) \
                        or (order_info['teacher_category'] == '985学霸' and teacher_info['teacher_type'] == 3):
                    if check_idle_time(order_info['classtime'][11:], teacher_info['tid'], idle_time, course_time):
                        state = True
            if mode == 'loose':
                if check_idle_time(order_info['classtime'][11:], teacher_info['tid'], idle_time, course_time):
                    state = True
        else:
            if mode == 'strict':
                if order_info['sex'] == teacher_info['sex']:
                    if (order_info['teacher_category'] == '一线老师' and teacher_info['teacher_type'] in (0, 1)) \
                            or (order_info['teacher_category'] == '985学霸' and teacher_info['teacher_type'] == 3):
                        if check_idle_time(order_info['classtime'][11:], teacher_info['tid'], idle_time, course_time):
                            state = True
            if mode == 'loose':
                if order_info['sex'] == teacher_info['sex']:
                    if check_idle_time(order_info['classtime'][11:], teacher_info['tid'], idle_time, course_time):
                        state = True
    return state


def check_idle_time(time, tid, idle_time, course_time):
    check_status = False
    idle_df = idle_time[idle_time.tid == tid]
    course_df = course_time[course_time.tid == tid]
    time_list = time_to_seconds(time)
    if idle_df.shape[0] > 0 and idle_df.iloc[0].at['start_time'] <= time_list[0] \
            and idle_df.iloc[0].at['end_time'] >= time_list[1]:
        if course_df.shape[0] > 0:
            conflict_time = course_df[~((course_df.day_start_time >= time_list[1]) |
                                        (course_df.day_end_time <= time_list[0]))]
            if conflict_time.shape[0] == 0:
                check_status = True
        else:
            check_status = True
    return check_status


def recievers_reset(recievers):
    for ind in range(4):
        recievers[ind]['status'], recievers[ind]['orders'] = 0, []
    return recievers


def teachers_unit_reset(units, units_spare):
    for uind in range(4):
        units[uind], units_spare[uind] = unit_adjuster(units[uind], units_spare[uind])
    return units, units_spare


def unit_adjuster(units, units_spare):
    units = pd.concat([units, units_spare], ignore_index=True)
    units = units.sort_values(by=['rate_type'])
    units = unit_inner_sort(units)
    units['order_nums'] = 0
    units_spare = pd.DataFrame(columns=list(units.columns))
    return units, units_spare


def teachers_unit_init(connect_config):
    teacher_info_sql = """
    select t1.tid, t2.sex, t2.bookVersion, t2.rate_type, t2.job_type,
    t2.teacher_type, substr(t2.identity_no,7,4) as birth_year,
    t2.school_age, t1.term, t1.subject from fudao_account.fudao_teacher as t1
    inner join fudao_account.fudao_teacher_info as t2 on t1.tid=t2.tid
    where t1.sign_status=1
    """
    teachers_init = pd.read_sql(teacher_info_sql, connect_config)
    teachers_init['bookVersion'] = teachers_init.bookVersion.apply(book_version_encode)
    teacher_unit_full = teachers_init[teachers_init.job_type == 2]
    teacher_unit_profession = teachers_init[teachers_init.job_type == 3]
    teacher_unit_base = teachers_init[teachers_init.job_type.isin([4, 5, 6])]
    teacher_unit_part_time = teachers_init[teachers_init.job_type == 1]
    teacher_unit_full_spare = None
    teacher_unit_profession_spare = None
    teacher_unit_base_spare = None
    teacher_unit_part_time_spare = None
    unit_list = [teacher_unit_full, teacher_unit_profession,
                 teacher_unit_base, teacher_unit_part_time]
    unit_spare_list = [teacher_unit_full_spare, teacher_unit_profession_spare,
                       teacher_unit_base_spare, teacher_unit_part_time_spare]
    unit_ind = 0
    while unit_ind < len(unit_list):
        unit_list[unit_ind] = teacher_rank_replace(unit_list[unit_ind])
        unit_list[unit_ind] = unit_list[unit_ind].drop_duplicates()
        unit_list[unit_ind] = unit_list[unit_ind].sort_values(by=['rate_type'])
        unit_list[unit_ind] = unit_inner_sort(unit_list[unit_ind])
        unit_list[unit_ind]['order_nums'] = 0
        unit_spare_list[unit_ind] = pd.DataFrame(columns=list(unit_list[unit_ind].columns))
        unit_ind += 1
    return unit_list, unit_spare_list


def unit_inner_sort(unit):
    job_type = unit.job_type.sample().iloc[0]
    if job_type == 2:
        sorted_list = []
        for t in ('A', 'B', 'C', 'D'):
            unit_tmp = unit[unit.rate_type == t]
            if unit_tmp.shape[0] > 0:
                unit_tmp = unit_tmp.sort_values(by=['term'], ascending=False)
                sorted_list.append(unit_tmp)
        sorted_unit = pd.concat(sorted_list)
    else:
        sorted_unit = unit
    sorted_unit = sorted_unit.reset_index(drop=True)
    return sorted_unit


def teacher_rank_replace(unit):
    if unit.job_type.sample().iloc[0] < 4:
        unit.rate_type = unit.rate_type.apply(lambda x:
                            'A' if x in (1, 2, 11) else 'B' if x == 3 else 'C' if x == 4
                            else 'D' if x == 5 else 'C')
    elif unit.job_type.sample().iloc[0] in (4, 5, 6):
        unit.rate_type = unit.rate_type.apply(lambda x:
                            'A' if x == 21 else 'B' if x == 22 else 'C' if x == 23 else 'D')
    else:
        unit.rate_type = unit.rate_type.apply(lambda x: 'C')
    return unit


def c_flow_guider(level_one_rec, level_two_rec, level_three_rec):
    if level_three_rec['status'] == 0:
        if level_one_rec['status'] == 0 and level_two_rec['status'] == 0:
            direction = direct_generator([0.3, 0.1, 0.6], ['A', 'B', 'C'])
        elif level_one_rec['status'] == 1 and level_two_rec['status'] == 0:
            direction = direct_generator([0.3, 0.7], ['B', 'C'])
        elif level_one_rec['status'] == 0 and level_two_rec['status'] == 1:
            direction = direct_generator([0.3, 0.7], ['A', 'C'])
        else:
            direction = 'C'
    else:
        if level_one_rec['status'] == 0 and level_two_rec['status'] == 0:
            direction = direct_generator([0.8, 0.2], ['A', 'B'])
        elif level_one_rec['status'] == 1 and level_two_rec['status'] == 0:
            direction = 'B'
        elif level_one_rec['status'] == 0 and level_two_rec['status'] == 1:
            direction = 'A'
        else:
            direction = 'C'
    return direction


def direct_generator(prob_list, directions):
    p = np.array(prob_list)
    direction = np.random.choice(directions, p=p.ravel())
    return direction


def month_begin(past_start_time=None):
    month_status = False
    past = past_start_time
    current = datetime.datetime.now()
    if (current-past).days > 0:
        if current.month == 2:
            if current.year/400 == 0 or (current.year/400 != 0 and current.year/4 == 0):
                 if current.day == 29:
                    month_status = True
            else:
                if current.day == 28:
                    month_status = True
        elif current.month in (1, 3, 5, 7, 8, 10, 12):
            if current.day == 31:
                month_status = True
        else:
            if current.day == 30:
                month_status = True
    return month_status


def get_order_level(order_score):
    if order_score is None:
        return 'D'
    if 0 <= order_score < 0.25:
        return 'D'
    elif 0.25 <= order_score < 0.5:
        return 'C'
    elif 0.5 <= order_score < 0.65:
        return 'B'
    else:
        return 'A'


if __name__ == '__main__':
    start_status = True
    count = 0
    month_start_time = datetime.datetime.now()
    xgb = XGBClassifier(
        learning_rate=0.1, n_estimators=120, max_depth=3, min_child_weight=7, gamma=0.7, subsample=0.8,
        colsample_bytree=0.8, objective='binary:logistic', nthread=4,
        scale_pos_weight=1, reg_alpha=0.01, seed=27)
    xgb.load_model('models/xgb.model')
    conn = tidbReader.pool.connection()
    engine = create_engine('mysql+pymysql://root:mysql@10.10.118.181:3306/match_result?charset=utf8')
    mysql_conn = engine.connect()
    order_inform, current_inform = [], None
    recievers, teachers_unit, teachers_unit_spare = [], [], []
    for _ in range(4):
        recievers.append({'status': 0, 'orders': []})
    while True:
        if month_begin(past_start_time=month_start_time):
            teachers_unit, teachers_unit_spare = teachers_unit_reset(teachers_unit, teachers_unit_spare)
            recievers = recievers_reset(recievers)
            month_start_time = datetime.datetime.now()
        msg = consumer.poll(5)
        if msg is None or msg.error():
            continue
        inform = msg.value()
        features = get_features(inform)
        p_features = process_features(features)
        order_score = xgb.predict_proba(p_features)
        order_level = get_order_level(order_score[0][1])
        conditions = get_conditions(inform)
        conditions['order_level'] = order_level
        if 'error_code' not in conditions.keys():
            try:
                recievers = order_scheduler(conditions, recievers)
                teacher_list, recievers, teachers_unit, teachers_unit_spare = teachers_allocator(recievers,
                                                                                                 teachers_unit,
                                                                                                 teachers_unit_spare,
                                                                                                 conn,
                                                                                                 start_status,
                                                                                                 month_start=month_start_time)
                start_status = False
                current_inform = [conditions['order_id'], conditions['order_level'], str(teacher_list), 0]
                print('当前订单获得推荐老师: %s' % str(teacher_list))
            except Exception as e:
                print('当前订单获取推荐老师异常，具体信息: %s' % str(e))
        else:
            current_inform = [conditions['order_id'], conditions['order_level'], '', 1]
        if current_inform is not None:
            order_inform.append(current_inform)
        if len(order_inform) == 20:
            order_inform_df = pd.DataFrame(order_inform,
                                           columns=['order_id', 'order_level', 'recommend_tids', 'status_code'])
            order_inform_df.to_sql('recommend-result', con=mysql_conn, if_exists='append')
            order_inform = []
        if inspect_timer(None, mode='day'):
            name_order = 0
            for unit in teachers_unit:
                if unit.shape[0] > 0:
                    table_name = 'unit_' + str(name_order)
                    unit[['tid', 'rate_type', 'order_nums']].to_sql(table_name, con=mysql_conn, if_exists='replace')
                name_order += 1
            name_order = 0
            for unit_spare in teachers_unit_spare:
                if unit_spare.shape[0] > 0:
                    table_name = 'unit_spare_' + str(name_order)
                    unit_spare[['tid', 'rate_type', 'order_nums']].to_sql(table_name, con=mysql_conn, if_exists='replace')
                name_order += 1
