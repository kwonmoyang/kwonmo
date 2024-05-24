import os
import time
import datetime
import decimal

import tabulate
import pandas as pd
import logging
import platform
import re

from dateutil import parser as date_parser
from typing import Union


# 전역변수 : datetime 변환용
G_STR_DATE_FORMAT = '%Y-%m-%d'  # date format
G_STR_WEEK_FORMAT = '%YW%W'  # Week format
G_STR_MONTH_FORMAT = '%YM%m'  # Month format


# 전역변수 : local 설정 변수
G_PROGRAM_NAME = None
G_IS_Local = None


# round 설정
G_CONTEXT = decimal.getcontext()
G_CONTEXT.rounding = decimal.ROUND_HALF_UP
G_DECIMAL = decimal.Decimal


tabulate.PRESERVE_WHITESPACE = True
tabulate_args = {
    'headers': 'keys',
    # 'tablefmt': 'grid',  # simple is the default format
    'disable_numparse': True,
    'showindex': True,
}


# class Singleton(type):
#     _instances = {}
#
#     def __call__(cls, *args, **kwargs):
#         if cls not in cls._instances:
#             cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
#         return cls._instances[cls]
#
#
# class G_MXLogger(metaclass=Singleton):
class G_MXLogger:
    """
    사용 예시:
      # 로거 객체 생성
      logger = G_MXLogger(p_py_name='template_training.py')

      # 시작 로그 출력
      logger.Start()

      # 스텝 종료 구분 (스텝번호, 설명)
      logger.Step(1, 'Step Description')  # <- Step 함수는 각 Step 이 끝나는 부분에서 호출
      logger.Step(2, '확정구간 생성 완료')  #    각 Step 이 성공적으로 끝났음을 확인
      logger.Step(3)

      # 종료 로그 출력
      logger.Finish()

      # 일반 문자열 출력 (로그 메시지, 로그 레벨)
      logger.Note('Debug 레벨 로그 예시', 10)
      logger.Note('Info 레벨 로그 예시', 20)

      # 데이터프레임 출력 (변수명, 데이터프레임명, format, row_num, condition)
      logger.PrintDF(df_fcst, 'Forecast', 1)  # 열 간격이 공백으로 출력
      logger.PrintDF(df_fcst, 'Forecast', 2)  # 열 간격이 세미콜론으로 구분되어 출력
      logger.PrintDF(df_fcst, 'Forecast', 1, 50)  # 50 줄만 출력 (기본: 5줄 출력)
      logger.PrintDF(df_fcst, 'Forecast', 1, 10, p_condition=['Item.DSM', 'Week'])  # ['Item.DSM', 'Week'] 컬럼만 출력
      logger.PrintDF(df_fcst, 'Forecast', 1, 10, p_condition=['Week', [12,]])  # 'Week' 컬럼에서 값이 12 인 데이터만 출력
      # 'Item.Item' 컬럼에서 값이 'Item.A1' 이거나 'Item.B2' 이면서,
      # 'Week' 컬럼에서 값이 12 또는 27 또는 29 인 컬럼 출력
      logger.PrintDF(df_fcst, 'Forecast', 1, 10, p_condition=['Item.Item', ['Item.A1', 'Item.B2'], 'Week', [12, 27, 29]])
    """
    def __init__(self, p_py_name):
        self._mx_logger = logging.getLogger('o9_logger')
        self._ts_start = time.time()  # TimeStamp (for Start_End)
        self._ts_step = time.time()  # TimeStamp (for Step)
        self.py_name = p_py_name  # python name
        self.prefix = '[NSCMLOG]'

        global G_PROGRAM_NAME, G_IS_Local
        G_PROGRAM_NAME = p_py_name
        G_IS_Local = gfn_get_isLocal()

    def get_level(self) -> int:
        return self._mx_logger.level

    def set_time_stamp(self, for_step: bool = False) -> None:
        if for_step:
            self._ts_step = time.time()
        else:
            self._ts_start = time.time()

    def get_time_stamp(self, for_step: bool = False) -> time:
        if for_step:
            return self._ts_step
        else:
            return self._ts_start

    def get_duration_time(self, for_step: bool = False) -> str:
        """
        time diff 계산,
        ts_step 리셋 (for step 인 경우,)
        :return: str
        """
        time_diff = time.time() - self.get_time_stamp(for_step)
        if for_step:
            self.set_time_stamp(for_step)

        return f'( {time_diff:.3f} sec)'

    # 데이터프레임 포매팅 (tabulate 라이브러리 이용하여 문자열로 변환)
    def df_formatter(self, df_in: pd.DataFrame, format_: int, row_num: int, condition) -> str:
        """
        컬럼 네임에 dtypes 추가
        데이터프레임 Shape 출력 ( oo Rows x oo Columns )
        :param df_in:
        :param format_: 1=tabulate, 2=csv
        :param row_num: 출력 행 [0:row_num]
        :param condition: 출력 데이터 필터
        :return: 문자열 ( 판다스 데이터프레임 -> str 로 변환 )
        """

        # 데이터프레임 정보 생성
        (row_, col_) = df_in.shape
        df_col_names = df_in.columns.values

        # 데이터프레임 shape 문자열 생성 ( oo Rows x oo Columns )
        shape_str = self.gen_shape_str(row=row_, col=col_)

        # 데이터프레임의 컬럼명에 dtype 을 붙여서 데이터프레임의 컬럼명을 변경하므로
        # 입력 데이터프레임의 복사본 생성이 필요 (동시에 row_num 까지만 사용, 불필요 데이터 제외 처리)
        copy_df = df_in[:row_num].copy()

        # 입력 파라미터 condition 이 있는지 확인
        if condition is not None:
            if isinstance(condition, list):
                # condition 이 'str' 로만 구성되어 있다면 특정 컬럼만 가져오는 필터로 작동
                if all(isinstance(c_, str) for c_ in condition):
                    # condition 의 각 요소에 대해, 데이터프레임의 col names 에 있으면,
                    cols = [c_ for c_ in condition if c_ in df_col_names]
                    # cols 컬럼만 가져온다. (row_num 까지만 사용하는 부분도 적용하기 위해 copy_df 에서 복사)
                    copy_df = copy_df[cols].copy()

                # 또는,
                # [col_name_1, [value_1_1, value_1_2, ...], col_name_2, [value_2_1, value_2_2, ...]] 형태인지 확인
                else:
                    cols = [foo for foo in condition[::2] if foo in df_col_names]
                    values = [foo for foo in condition[1::2]]
                    if all((len(cols) == len(values),  # condition 에서 col_name 과 [values..] 가 쌍이 맞는지 확인,
                            # (isinstance(col, str) for col in cols),  # cols 생성하면서 이미 확인 되었으므로 주석,
                            (isinstance(val_, list) for val_ in values))):  # .isin 에 list 가 필요하므로 확인,
                        # df_in.loc[df_in[ ].isin([ ])]
                        cdt_ = ' & '.join(f'df_in[cols[{idx}]].isin(values[{idx}])' for idx in range(len(cols)))
                        copy_df = eval(f'df_in.loc[{cdt_}]')

        # 데이터프레임을 문자열로 변환
        msg_ = ''  # place holder
        if format_ == 1:
            # 컬럼 네임에 dtypes 추가
            copy_df = self.tabulate_column_w_dtypes(copy_df)
            # tabulate 라이브러리로 출력 (+개행)
            msg_ = f'{tabulate.tabulate(copy_df, **tabulate_args)}\n'
        elif format_ == 2:
            # 판다스의 .to_csv() 를 이용해서 출력
            msg_ = copy_df.to_csv(
                sep=';',  # 구분자 (default ',')
                line_terminator='\n'  # 개행문자 (optional, \n for linux, \r\n for Windows, i.e.)
            )
            # 위의 to_csv() 에서 출력된 문자열을 dtypes 가 추가된 문자열로 바꾼다.
            msg_ = self.add_dtypes_to_csv_header(copy_df, msg_)

        # return Msg.
        return f'{msg_}DataFrame Shape: {shape_str}'

    # 'oo Rows x oo Columns' 문자열 생성
    @staticmethod
    def gen_shape_str(row: int = 0, col: int = 0) -> str:
        row_str = f'{row:,} Row{"s"[:row ^ 1]}'  # row 가 1이 아니면 s 를 붙여서 복수형으로 표현
        col_str = f'{col} Column{"s"[:col ^ 1]}'
        shape_str = f'{row_str} x {col_str}'

        return shape_str

    # DataFrame 의 컬럼 네임 변경 (+ dtypes)
    @staticmethod
    def tabulate_column_w_dtypes(copy_df: pd.DataFrame) -> pd.DataFrame:
        """
        입력받은 데이터프레임의 컬럼 네임에 dtypes 를 붙여서 리턴
        :param copy_df: 입력된 데이터프레임 (원본에 영향이 없도록 복사본이 생성되어 입력됨)
                        -> tabulate 로 출력하면서 column name 이 dtypes 가 추가된 형태로 나와야 하므로
                           본 함수에서 리턴하는 데이터프레임이 로그 출력에 사용될 데이터프레임이 된다.
        :return: 컬럼 네임에 dtypes 가 추가된 데이터프레임
        """
        col_names = copy_df.columns.values
        if len(col_names):
            dtypes_ = [_.name for _ in copy_df.dtypes.tolist()]  # dtypes 를 가져와서 list 로 변환
            column_w_dtypes = [f'{_[0]}\n({_[1]})' for _ in zip(col_names, dtypes_)]  # 결합 (column name + '\n' + dtype)
            new_col_names = dict(zip(col_names, column_w_dtypes))
            copy_df = copy_df.rename(new_col_names, axis='columns', inplace=False)  # 데이터프레임에서 컬럼 네임만 교체
        return copy_df

    # csv 출력 형식일 때 dtypes 문자열 추가
    @staticmethod
    def add_dtypes_to_csv_header(copy_df: pd.DataFrame, to_csv_msg: str) -> str:
        """
        pd.to_csv() 의 결과 문자열에 dtypes 추가
        :param copy_df: dtypes 를 가져올 데이터프레임
        :param to_csv_msg: 데이터프레임을 출력한 문자열 (세미콜론 구분자)
        :return: 입력으로 받은 to_csv_msg 의 첫 번째 개행문자 다음에 dtypes 를 추가한 결과물
        """
        # 첫 번째 개행문자를 기준으로 헤더와 그 나머지 부분을 분리
        (head_, tail_) = ('', '')  # place holder
        if len(to_csv_msg) and ('\n' in to_csv_msg):
            (head_, tail_) = to_csv_msg.split('\n', maxsplit=1)

        # dtypes 를 세미콜론으로 이어붙이고 리턴 msg 를 만든다
        len_col_names = len(copy_df.columns.values)
        if len_col_names and len(head_):
            dtypes_ = [f'({_.name})' for _ in copy_df.dtypes.tolist()]  # dtypes 를 가져와서
            dtypes_ = ';'.join(dtypes_)  # 세미콜론으로 이어붙인다
            if head_.count(';') == len_col_names:  # 구분자인 세미콜론의 개수는 컬럼명 개수보다 하나 적어야 하지만 만약 같다면,
                dtypes_ = f';{dtypes_}'  # 데이터프레임 Index Name 의 빈자리라고 보고 세미콜론을 맨 앞에 하나 추가
            # dtypes 추가하여 msg 다시 생성
            to_csv_msg = f'{head_}\n{dtypes_}\n{tail_}'

        return to_csv_msg

    def debug(self, msg, *args, **kwargs):
        self._mx_logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self._mx_logger.info(msg, *args, **kwargs)

    # def warning(self, msg, *args, **kwargs):
    #     self.mx_logger.warning(msg, *args, **kwargs)
    #
    # def error(self, msg, *args, **kwargs):
    #     self.mx_logger.error(msg, *args, **kwargs)
    #
    # def critical(self, msg, *args, **kwargs):
    #     self.mx_logger.critical(msg, *args, **kwargs)

    def Start(self):
        self.set_time_stamp()
        self._emit_info_level('Start', with_duration=False)

    def Step(self, p_step_no: int, p_step_desc: str = ''):
        self._emit_info_level(f'Step.{int(float(p_step_no)):03};{p_step_desc}', for_step=True)

    def Note(self, p_note: str, p_log_level: int = 10):
        if p_log_level == 20:
            self._emit_info_level(p_note, with_duration=False)
        else:
            self._emit_debug_level(p_note)

    def PrintDF(self, p_df_, p_df_name: str, p_format: int = 1, p_row_num: int = 5, p_condition: list = None):
        msg_ = self.df_formatter(p_df_, p_format, p_row_num, p_condition)
        # send log
        if len(msg_):
            self._emit_debug_level(f'{p_df_name}\n{msg_}')

    def Error(self):
        self._emit_info_level('Error')

    def Finish(self, p_df_=None):
        self._emit_info_level('Finish')
        # local file 출력
        if G_IS_Local is True and p_df_ is not None:
            gfn_get_down_csv_file(p_dataframe=p_df_, p_dir='output')

    def _emit_info_level(self, msg: str, with_duration: bool = True, for_step: bool = False):
        msg_ = f'{self.prefix};{self.py_name};{msg}'
        if with_duration:
            dur_time = self.get_duration_time(for_step)
            msg_ += f';{dur_time}'

        self._mx_logger.info(msg_)

    def _emit_debug_level(self, msg: str):
        self._mx_logger.debug(f'{self.py_name};{msg}')



###########################################################
# log_level object
###########################################################
class G_log_level:
    """
    log_level.debug
    """
    @staticmethod
    def debug(): return 10

    @staticmethod
    def info(): return 20

    @staticmethod
    def warning(): return 30

    @staticmethod
    def error(): return 40

    @staticmethod
    def critital(): return 40


###########################################################
# Local 개발 환경 설정 함수 3개
#       get_isLocal
#       set_local_logfile
#       get_down_csv_file
###########################################################
def gfn_get_isLocal() -> bool:
    """
    Local 구분

    :return: bool
    """
    bool_local = False
    if platform.system() not in ['Linux']:
        bool_local = True
    return bool_local


def gfn_set_local_logfile() -> None:
    """
    Local log file 설정

    :return:
    """
    if G_IS_Local is True:
        logging.getLogger().setLevel(logging.DEBUG)

    if G_IS_Local is True and G_PROGRAM_NAME is not None:
        log_file_name = G_PROGRAM_NAME.replace('py', 'log')
        log_file_name = f'log/{log_file_name}'
        if os.path.exists(log_file_name):
            os.remove(log_file_name)
        file_handler = logging.FileHandler(log_file_name, encoding='UTF-8')
        logging.getLogger('o9_logger').addHandler(file_handler)


def gfn_get_down_csv_file(p_dataframe: pd.DataFrame, p_dir='output') -> None:
    """
    Local에서 최종 Out Dataframe -> .csv 파일로 출력

    :param p_dataframe:
    :param p_dir:
    :return:
    """
    if G_IS_Local is True:
        csv_date = datetime.datetime.now().strftime('%Y%m%d_%H_%M')
        csv_out_filename = f'{p_dir}/{csv_date}_out_{G_PROGRAM_NAME.replace(".py", "")}.csv'
        p_dataframe.to_csv(csv_out_filename, index=False)


###########################################################
# datetime 변환 함수 10개 예시
#          TimeDimToDate_W  : TimeDimToDate_W('2024W01') -> datetime(2024-01-01 00:00:00)
#          TimeDateToChar_W : TimeDateToChar_W(2024.01.01 03:09:09) -> '2024W01'
#          TimeDimToDate_M  : TimeDimToDate_M('2024M01') -> datetime(2024-01-01 00:00:00)
#          TimeDateToChar_M : TimeDateToChar_M(2024.01.01 03:09:09) -> '2024M01'
#          to_date          : to_date('2024-W01', '%Y-W%W') -> datetime(2024-01-01 00:00:00)
#          to_char          : to_char(datetime(2024-01-01 00:00:00), '%Y-W%W') -> 2024-W01
#          is_date_parsing
#          is_date_matching
#          get_df_mst_week  : df_mst_week2 = common.get_df_mst_week(p_frist_week='202401', p_duration_week=30, p_in_out_week_format='%Y%W')
#          get_df_mst_week_from_date : df_mst_week5 = common.get_df_mst_week_from_date(p_frist_day='2024-01-01', p_duration_week=30, p_in_out_week_format='%Y%W', p_in_out_day_format='%Y-%m-%d')
###########################################################
def gfn_TimeDimToDate_W(p_str_datetype: str, p_week_day=1) -> datetime:
    """
    string -> datetime
    ex) TimeDimToDate_W('2024W01') -> datetime(2024-01-01 00:00:00)

    :param p_str_datetype:
    :param p_week_day:
    :return:
    """
    str_msg = ''
    if r'%W' in G_STR_WEEK_FORMAT and gfn_is_date_matching(p_str_datetype, G_STR_WEEK_FORMAT):
        year, week = None, None
        all_char = re.sub(r'[^0-9]', '', p_str_datetype)
        if len(all_char) == 6:
            year = int(all_char[:4])
            week = int(all_char[4:])
        elif len(all_char) == 5:
            year = int(all_char[:4])
            week = int(all_char[-1:])
        else:
            str_msg = f'''Error : week format string not matching
            common function : TimeDimToDate_W -> is_date_matching
            param    : ({p_str_datetype}, {p_week_day})
            '''
            raise Exception(str_msg)
        # return datetime.datetime.fromisocalendar(year, week, p_week_day)
        return datetime.datetime.strptime(f"{year:04d}{week:02d}{p_week_day:d}", "%G%V%u")  # .date()
    else:
        str_msg = f'''Error : week format string not matching
        common function : TimeDimToDate_W -> is_date_matching
        param    : ({p_str_datetype}, {p_week_day})
        format   : {G_STR_WEEK_FORMAT}
        '''
        raise Exception(str_msg)


def gfn_TimeDateToChar_W(p_datetime: datetime) -> str:
    """
    datetime -> string
    ex) TimeDateToChar_W(2024.01.01 03:09:09) -> '2024W01'

    :param p_datetime:
    :return:
    """
    str_msg = ''
    if r'%W' in G_STR_WEEK_FORMAT:
        # year = str(p_datetime.isocalendar().year)
        # week = str(p_datetime.isocalendar().week).zfill(2)
        year = str(p_datetime.isocalendar()[0])
        week = str(p_datetime.isocalendar()[1]).zfill(2)
        return G_STR_WEEK_FORMAT.replace('%Y', year).replace('%W', week)
    else:
        str_msg = f'''Error : format string not matching
        common function : TimeDateToChar_W -> is_date_matching
        param    : ({p_datetime})
        format   : {G_STR_WEEK_FORMAT}
        '''
        raise Exception(str_msg)


def gfn_TimeDimToDate_M(p_str_datetype: str) -> datetime:
    """
    string -> datetime
    ex) TimeDimToDate_M('2024M01') -> datetime(2024-01-01 00:00:00)

    :param p_str_datetype:
    :return:
    """
    str_msg = ''
    if r'%m' in G_STR_MONTH_FORMAT and gfn_is_date_matching(p_str_datetype, G_STR_MONTH_FORMAT):
        year, month = None, None
        all_char = re.sub(r'[^0-9]', '', p_str_datetype)
        if len(all_char) == 6:
            year = all_char[:4]
            month = all_char[4:]
        elif len(all_char) == 5:
            year = all_char[:4]
            month = all_char[-1:]
        else:
            str_msg = f'''Error : month format string not matching
            common function : TimeDimToDate_M -> is_date_matching
            param    : ({p_str_datetype})
            '''
            raise Exception(str_msg)
        str_datetype = '-'.join([year, month, '01'])
        return datetime.datetime.strptime(str_datetype, '%Y-%m-%d')
    else:
        str_msg = f'''Error : month format string not matching
        common function : TimeDimToDate_M -> is_date_matching
        param    : ({p_str_datetype})
        format   : {G_STR_MONTH_FORMAT}
        '''
        raise Exception(str_msg)


def gfn_TimeDateToChar_M(p_datetime: datetime) -> str:
    """
    datetime -> string
    ex) TimeDateToChar_M(2024.01.01 03:09:09) -> '2024M01'

    :param p_datetime:
    :return:
    """
    str_msg = ''
    if r'%m' in G_STR_MONTH_FORMAT:
        year = p_datetime.strftime('%Y')
        month = p_datetime.strftime('%m')
        return G_STR_MONTH_FORMAT.replace('%Y', year).replace('%m', month)
    else:
        str_msg = f'''Error : month format string not matching
        common function : TimeDateToChar_M -> is_date_matching
        param    : ({p_datetime})
        format   : {G_STR_MONTH_FORMAT}
        '''
        raise Exception(str_msg)


def gfn_to_date(p_str_datetype: str, p_format: str, p_week_day=1, p_day_delta=0) -> datetime:
    """
    string -> datetime
    ex) to_date('2024-W01', '%Y-W%W') -> datetime(2024-01-01 00:00:00)
        to_date('2024-M01', '%Y-M%m') -> datetime(2024-01-01 00:00:00)
        to_date('20240101', '%Y%m%d') -> datetime(2024-01-01 00:00:00)
        to_date('2024.01.01', '%Y.%m.%d') -> datetime(2024-01-01 00:00:00)
        to_date('2024.01.01 03:09:09', '%Y.%m.%d %H:%M:%S') -> datetime(2024-01-01 03:09:09)

    :param p_str_datetype:
    :param p_format:
    :param p_week_day:
    :param p_day_delta:
    :return:
    """
    result = None
    str_msg = ''
    if r'%W' in p_format and gfn_is_date_matching(p_str_datetype, p_format):
        year, week = None, None
        all_char = re.sub(r'[^0-9]', '', p_str_datetype)
        if len(all_char) == 6:
            year = int(all_char[:4])
            week = int(all_char[4:])
        elif len(all_char) == 5:
            year = int(all_char[:4])
            week = int(all_char[-1:])
        else:
            str_msg = f'''Error : week format string not matching
            common function : to_date -> is_date_matching
            param    : ({p_str_datetype}, {p_format}, {p_week_day})
            '''
            raise Exception(str_msg)

        # result = datetime.datetime.fromisocalendar(year, week, p_week_day)
        return datetime.datetime.strptime(f"{year:04d}{week:02d}{p_week_day:d}", "%G%V%u")  # .date()

    elif r'%m' in p_format and r'%d' not in p_format and gfn_is_date_matching(p_str_datetype, p_format):
        year, month = None, None
        all_char = re.sub(r'[^0-9]', '', p_str_datetype)
        if len(all_char) == 6:
            year = all_char[:4]
            month = all_char[4:]
        elif len(all_char) == 5:
            year = all_char[:4]
            month = all_char[-1:]
        else:
            str_msg = f'''Error : month format string not matching
            common function : to_date -> is_date_matching
            param    : ({p_str_datetype})
            '''
            raise Exception(str_msg)
        str_datetype = '-'.join([year, month, '01'])
        return datetime.datetime.strptime(str_datetype, '%Y-%m-%d')
    else:
        if gfn_is_date_parsing(p_str_datetype):
            if gfn_is_date_matching(p_date_str=p_str_datetype, p_format=p_format):
                result = datetime.datetime.strptime(p_str_datetype, p_format)
            else:
                str_msg = f'''Error : format string not matching
                common function : to_date -> is_date_matching
                param    : ({p_str_datetype}, {p_format}, {p_week_day})
                '''
                raise Exception(str_msg)
        else:
            str_msg = f'''Error : format string not parsing
            common function : to_date -> is_date_parsing
            param    : ({p_str_datetype}, {p_format}, {p_week_day})
            '''
            raise Exception(str_msg)

    if p_day_delta == 0:
        return result
    else:
        return result + datetime.timedelta(days=p_day_delta)


def gfn_to_char(p_datetime: datetime, p_format: str, p_day_delta=0) -> str:
    """
    string -> datetime
    ex) to_char(datetime(2024-01-01 00:00:00), '%Y-W%W') -> 2024-W01
        to_char(datetime(2024-01-01 00:00:00), '%Y-M%m') -> 2024-M01
        to_char(datetime(2024-01-01 00:00:00), '%Y%m%d') -> 20240101
        to_date(datetime(2024-01-01 00:00:00), '%Y.%m.%d') -> 2024.01.01
        to_date(datetime(2024-01-01 03:09:09), '%Y.%m.%d %H:%M:%S') -> 2024.01.01 03:09:09

    :param p_datetime:
    :param p_format:
    :param p_day_delta:
    :return:
    """
    result = None
    str_msg = ''
    if p_day_delta != 0:
        p_datetime = p_datetime + datetime.timedelta(days=p_day_delta)

    if r'%W' in p_format:
        # year = str(p_datetime.isocalendar().year)
        # week = str(p_datetime.isocalendar().week).zfill(2)
        year = str(p_datetime.isocalendar()[0])
        week = str(p_datetime.isocalendar()[1]).zfill(2)
        result = p_format.replace('%Y', year).replace('%W', week)
    elif r'%m' in p_format and r'%d' not in p_format:
        year = p_datetime.strftime('%Y')
        month = p_datetime.strftime('%m')
        return p_format.replace('%Y', year).replace('%m', month)
    else:
        if gfn_is_date_matching(p_date_str=p_datetime, p_format=p_format):
            result = datetime.datetime.strftime(p_datetime, p_format)
        else:
            str_msg = f'''Error : format string not matching
            common function : to_date -> is_date_matching
            param    : ({p_datetime}, {p_format}, {p_day_delta})
            '''
            raise Exception(str_msg)

    return result


def gfn_is_date_parsing(p_date_str: str) -> bool:
    try:
        return bool(date_parser.parse(p_date_str))
    except ValueError:
        return False


def gfn_is_date_matching(p_date_str: Union[str, datetime.datetime], p_format) -> bool:
    try:
        if isinstance(p_date_str, str):
            return bool(datetime.datetime.strptime(p_date_str, p_format))
        else:
            return bool(datetime.datetime.strftime(p_date_str, p_format))
    except ValueError:
        return False


def gfn_get_df_mst_week(p_frist_week: str, p_duration_week=None, p_in_out_week_format=G_STR_WEEK_FORMAT, p_duration_day=None) -> pd.DataFrame:
    """
    week 기준 dataframe 생성
    df_mst_week = common.get_df_mst_week(p_frist_week='202401', p_duration_week=30)
    df_mst_week = common.get_df_mst_week(p_frist_week='202401', p_duration_week=30, p_in_out_week_format='%Y%W')
    df_mst_week = common.get_df_mst_week(p_frist_week='202401', p_duration_day=365)

    :param p_frist_week:
    :param p_duration_week:
    :param p_in_out_week_format:
    :param p_duration_day:
    :return:
    """
    date_start = gfn_to_date(p_str_datetype=p_frist_week, p_format=p_in_out_week_format)

    list_loop = []
    if p_duration_day is None:
        for i in range(0, p_duration_week):
            list_loop.append(i*7)
    else:
        for i in range(0, p_duration_day, 7):
            list_loop.append(i)

    list_result_week = []
    for i, value in enumerate(list_loop):
        str_week = gfn_to_char(p_datetime=date_start, p_format=p_in_out_week_format, p_day_delta=value)
        list_result_week.append(str_week)
    return pd.DataFrame(list_result_week, columns=['week'])


def gfn_get_df_mst_week_from_date(p_frist_day: Union[str, datetime.datetime], p_duration_week=None,
                              p_in_out_week_format=G_STR_WEEK_FORMAT, p_in_out_day_format=G_STR_DATE_FORMAT, p_duration_day=None) -> pd.DataFrame:
    """
    week 기준 dataframe 생성
    df_mst_week = common.get_df_mst_week_from_date(p_frist_day='20240101', p_duration_week=30)
    df_mst_week = common.get_df_mst_week_from_date(p_frist_day='2024-01-01', p_duration_week=30, p_in_out_week_format='%Y%W', p_in_out_day_format='%Y-%m-%d')
    df_mst_week = common.get_df_mst_week_from_date(p_frist_day=datetime.datetime.now(), p_duration_day=365)

    :param p_frist_day:
    :param p_duration_week:
    :param p_in_out_week_format:
    :param p_in_out_day_format:
    :param p_duration_day:
    :return:
    """
    if isinstance(p_frist_day, str):
        date_start = gfn_to_date(p_str_datetype=p_frist_day, p_format=p_in_out_day_format)
    else:
        date_start = p_frist_day

    list_loop = []
    if p_duration_day is None:
        for i in range(0, p_duration_week):
            list_loop.append(i * 7)
    else:
        for i in range(0, p_duration_day, 7):
            list_loop.append(i)

    list_result_week = []
    for i, value in enumerate(list_loop):
        str_week = gfn_to_char(p_datetime=date_start, p_format=p_in_out_week_format, p_day_delta=value)
        list_result_week.append(str_week)
    return pd.DataFrame(list_result_week, columns=['week'])


###########################################################
# round 함수
#       gfn_get_round
###########################################################
def gfn_get_round(p_float: Union[str, float], p_decimal=0) -> decimal.Decimal:
    """
    # round 예시
    print(common.get_round(2.999999999999975, p_decimal=8))
    df['col1 -> common apply'] = df['col1'].apply(common.get_round)
    df['col1 -> common map']   = df['col1'].map(common.get_round)
    df['col2 -> common round 2'] = df['col2'].apply(common.get_round, p_decimal=2)

    :param p_float:
    :param p_decimal:
    :return:
    """
    if isinstance(p_float, float):
        p_float = str(p_float)
    return round(G_DECIMAL(p_float), p_decimal)


###########################################################
# 최빈값 설정 함수
#       gfn_set_rep
###########################################################
def gfn_set_rep(p_df: pd.DataFrame, p_list_key: list, p_str_rep_column: str) -> tuple:
    """
    # 최빈값 설정 예시 : return tuple
    (최빈값을 반영한 dataframe, 최빈값 master dataframe)
    df_in_rep, df_mst_rep = common.set_rep(df_in_rep, p_list_key=['DC'], p_str_rep_column='REP_DC')

    :param p_df:
    :param p_list_key:
    :param p_str_rep_column:
    :return:
    """
    list_all = p_list_key + [p_str_rep_column]
    list_sort = p_list_key + ['set_rep.count']
    list_ascending = [True for i in list_sort if i not in ['set_rep.count']] + [False]

    _df_rep = p_df[list_all].copy()

    _df_rep['set_rep.count'] = 1
    _df_rep = _df_rep.groupby(list_all)['set_rep.count'].sum().reset_index()

    _df_rep = _df_rep.sort_values(by=list_sort, ascending=list_ascending).reset_index(drop=True)

    _df_rep['set_rep.rank'] = _df_rep.groupby(p_list_key)['set_rep.count'].rank(method='min', ascending=False)
    _df_rep = _df_rep.loc[_df_rep['set_rep.rank'] == 1]

    _df_rep.drop_duplicates(p_list_key, keep='first', inplace=True)
    _df_rep = _df_rep.drop(['set_rep.count', 'set_rep.rank'], axis=1)  # column 제거

    p_df = p_df.drop([p_str_rep_column], axis=1)  # column 제거
    return (pd.merge(p_df, _df_rep, how='left', on=p_list_key), _df_rep.copy())
