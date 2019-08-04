import logging
# from common.utils import date_utils, html_table
import datetime

def eval_arg(arg_value, arg_name=''):
    """
    Try to type a str argument, detecting booleans, integers. If the arg name ends with "_list", try to make a list
    out of it, splitting around ",".
    """
    if arg_name.lower().endswith('_list') and isinstance(arg_value, str):
        return [eval_arg(cell) for cell in arg_value.split(',')]
    if not isinstance(arg_value, str):
        return arg_value
    if arg_value.lower() in ['true', 'false']:
        return eval(arg_value.capitalize())
    if arg_value.lstrip('-').isdigit():
        return int(arg_value)
    if arg_value.replace('.', '', 1).isdigit():
        return float(arg_value)
    return arg_value


class BaseTask:
    """
    """
    defaults = {}
    defaults['date_offset'] = 0
    defaults['timezone'] = 'America/New York'
    defaults['calendar'] = ''
    defaults['datetime_fmt'] = '%Y.%m.%d %H:%M:%S'
    defaults['date_fmt'] = '%Y.%m.%d'

    def __init__(self, **kwargs):
        logging.info('Running ' + str(self.__class__.__name__) + ' with parameters: ' + str(kwargs))
        print('Running ' + str(self.__class__.__name__) + ' with parameters: ' + str(kwargs))
        self.kwargs = kwargs
        self.init_attributes()

    def init_attributes(self):
        """
            Initialize certain attributes to default values.
            Set all kwargs as attributes.
        """
        # Set default values
        for key, value in self.defaults.items():
            setattr(self, key, value)

        # Parse all arguments in kwargs
        for key, value in self.kwargs.items():
            parsed_value = eval_arg(value, key)
            logging.info('Setting ' + str(type(parsed_value)) + ' self.' + str(key) + ' = ' + str(parsed_value))
            setattr(self, key, parsed_value)

        # self.today = date_utils.get_datetime_from_timezone(self.date_offset, self.timezone)
        self.today = datetime.datetime.today()

    def iteration(self):
        """"Yout code here": method to implement"""
        logging.info('To implement: self.iteration()')
        return True

    def run(self):
        """"""
        # self.html_report = html_table.HtmlTable()
        return self.iteration()

# params = {}                                                                                                                          ## **params -> (a='a', b='b', c='c')
# params['to_table_name'] = 'matable'                                                                                                  ## **params -> (a='a', b='b', c='c')
# params['date_offset'] = '0'                                                                                                 ## **params -> (a='a', b='b', c='c')
# params['a3'] = '0'                                                                                                  ##  *params -> (a, b, c)
#
# task = BaseTask(**params)
# task = BaseTask(to_table_name='', date_offset=0)
#
# params = {}
# import os
# params['MAIN'] = os.getcwd()
#
# obj = BaseTask(**params)
# obj.run()