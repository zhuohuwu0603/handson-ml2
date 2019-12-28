#!/usr/bin/env python
# coding: utf-8

# This notebook was prepared by [Donne Martin](http://donnemartin.com). Source and license info is on [GitHub](https://github.com/donnemartin/data-science-ipython-notebooks).

# # Python Hadoop MapReduce: Analyzing AWS S3 Bucket Logs with mrjob
# 
# * [Introduction](#Introduction)
# * [Setup](#Setup)
# * [Processing S3 Logs](#Processing-S3-Logs)
# * [Running Amazon Elastic MapReduce Jobs](#Running-Amazon-Elastic-MapReduce-Jobs)
# * [Unit Testing S3 Logs](#Unit-Testing-S3-Logs)
# * [Running S3 Logs Unit Test](#Running-S3-Logs-Unit-Test)
# * [Sample Config File](#Sample-Config-File)

# ## Introduction
# 
# [mrjob](https://pythonhosted.org/mrjob/) lets you write MapReduce jobs in Python 2.5+ and run them on several platforms. You can:
# 
# * Write multi-step MapReduce jobs in pure Python
# * Test on your local machine
# * Run on a Hadoop cluster
# * Run in the cloud using Amazon Elastic MapReduce (EMR)

# ## Setup
# 
# From PyPI:
# 
# ``pip install mrjob``
# 
# From source:
# 
# ``python setup.py install``
# 
# See [Sample Config File](#Sample-Config-File) section for additional config details.

# ## Processing S3 Logs
# 
# Sample mrjob code that processes log files on Amazon S3 based on the [S3 logging format](http://docs.aws.amazon.com/AmazonS3/latest/dev/LogFormat.html):

# In[ ]:


# get_ipython().run_cell_magic('file', 'mr_s3_log_parser.py', '\nimport time\nfrom mrjob.job import MRJob\nfrom mrjob.protocol import RawValueProtocol, ReprProtocol\nimport re\n\n\nclass MrS3LogParser(MRJob):\n    """Parses the logs from S3 based on the S3 logging format:\n    http://docs.aws.amazon.com/AmazonS3/latest/dev/LogFormat.html\n    \n    Aggregates a user\'s daily requests by user agent and operation\n    \n    Outputs date_time, requester, user_agent, operation, count\n    """\n\n    LOGPATS  = r\'(\\S+) (\\S+) \\[(.*?)\\] (\\S+) (\\S+) \' \\\n               r\'(\\S+) (\\S+) (\\S+) ("([^"]+)"|-) \' \\\n               r\'(\\S+) (\\S+) (\\S+) (\\S+) (\\S+) (\\S+) \' \\\n               r\'("([^"]+)"|-) ("([^"]+)"|-)\'\n    NUM_ENTRIES_PER_LINE = 17\n    logpat = re.compile(LOGPATS)\n\n    (S3_LOG_BUCKET_OWNER, \n     S3_LOG_BUCKET, \n     S3_LOG_DATE_TIME,\n     S3_LOG_IP, \n     S3_LOG_REQUESTER_ID, \n     S3_LOG_REQUEST_ID,\n     S3_LOG_OPERATION, \n     S3_LOG_KEY, \n     S3_LOG_HTTP_METHOD,\n     S3_LOG_HTTP_STATUS, \n     S3_LOG_S3_ERROR, \n     S3_LOG_BYTES_SENT,\n     S3_LOG_OBJECT_SIZE, \n     S3_LOG_TOTAL_TIME, \n     S3_LOG_TURN_AROUND_TIME,\n     S3_LOG_REFERER, \n     S3_LOG_USER_AGENT) = range(NUM_ENTRIES_PER_LINE)\n\n    DELIMITER = \'\\t\'\n\n    # We use RawValueProtocol for input to be format agnostic\n    # and avoid any type of parsing errors\n    INPUT_PROTOCOL = RawValueProtocol\n\n    # We use RawValueProtocol for output so we can output raw lines\n    # instead of (k, v) pairs\n    OUTPUT_PROTOCOL = RawValueProtocol\n\n    # Encode the intermediate records using repr() instead of JSON, so the\n    # record doesn\'t get Unicode-encoded\n    INTERNAL_PROTOCOL = ReprProtocol\n\n    def clean_date_time_zone(self, raw_date_time_zone):\n        """Converts entry 22/Jul/2013:21:04:17 +0000 to the format\n        \'YYYY-MM-DD HH:MM:SS\' which is more suitable for loading into\n        a database such as Redshift or RDS\n\n        Note: requires the chars "[ ]" to be stripped prior to input\n        Returns the converted datetime annd timezone\n        or None for both values if failed\n\n        TODO: Needs to combine timezone with date as one field\n        """\n        date_time = None\n        time_zone_parsed = None\n\n        # TODO: Probably cleaner to parse this with a regex\n        date_parsed = raw_date_time_zone[:raw_date_time_zone.find(":")]\n        time_parsed = raw_date_time_zone[raw_date_time_zone.find(":") + 1:\n                                         raw_date_time_zone.find("+") - 1]\n        time_zone_parsed = raw_date_time_zone[raw_date_time_zone.find("+"):]\n\n        try:\n            date_struct = time.strptime(date_parsed, "%d/%b/%Y")\n            converted_date = time.strftime("%Y-%m-%d", date_struct)\n            date_time = converted_date + " " + time_parsed\n\n        # Throws a ValueError exception if the operation fails that is\n        # caught by the calling function and is handled appropriately\n        except ValueError as error:\n            raise ValueError(error)\n        else:\n            return converted_date, date_time, time_zone_parsed\n\n    def mapper(self, _, line):\n        line = line.strip()\n        match = self.logpat.search(line)\n\n        date_time = None\n        requester = None\n        user_agent = None\n        operation = None\n\n        try:\n            for n in range(self.NUM_ENTRIES_PER_LINE):\n                group = match.group(1 + n)\n\n                if n == self.S3_LOG_DATE_TIME:\n                    date, date_time, time_zone_parsed = \\\n                        self.clean_date_time_zone(group)\n                    # Leave the following line of code if \n                    # you want to aggregate by date\n                    date_time = date + " 00:00:00"\n                elif n == self.S3_LOG_REQUESTER_ID:\n                    requester = group\n                elif n == self.S3_LOG_USER_AGENT:\n                    user_agent = group\n                elif n == self.S3_LOG_OPERATION:\n                    operation = group\n                else:\n                    pass\n\n        except Exception:\n            yield (("Error while parsing line: %s", line), 1)\n        else:\n            yield ((date_time, requester, user_agent, operation), 1)\n\n    def reducer(self, key, values):\n        output = list(key)\n        output = self.DELIMITER.join(output) + \\\n                 self.DELIMITER + \\\n                 str(sum(values))\n\n        yield None, output\n\n    def steps(self):\n        return [\n            self.mr(mapper=self.mapper,\n                    reducer=self.reducer)\n        ]\n\n\nif __name__ == \'__main__\':\n    MrS3LogParser.run()')


# ## Running Amazon Elastic MapReduce Jobs

# Run an Amazon Elastic MapReduce (EMR) job on the given input (must be a flat file hierarchy), placing the results in the output (output directory must not exist):

# In[ ]:


# get_ipython().system('python mr_s3_log_parser.py -r emr s3://bucket-source/ --output-dir=s3://bucket-dest/')


# Run a MapReduce job locally on the specified input file, sending the results to the specified output file:

# In[ ]:


# get_ipython().system('python mr_s3_log_parser.py input_data.txt > output_data.txt')


# ## Unit Testing S3 Logs

# Accompanying unit test:

# In[ ]:


# get_ipython().run_cell_magic('file', 'test_mr_s3_log_parser.py', '\nfrom StringIO import StringIO\nimport unittest2 as unittest\nfrom mr_s3_log_parser import MrS3LogParser\n\n\nclass MrTestsUtil:\n\n    def run_mr_sandbox(self, mr_job, stdin):\n        # inline runs the job in the same process so small jobs tend to\n        # run faster and stack traces are simpler\n        # --no-conf prevents options from local mrjob.conf from polluting\n        # the testing environment\n        # "-" reads from standard in\n        mr_job.sandbox(stdin=stdin)\n\n        # make_runner ensures job cleanup is performed regardless of\n        # success or failure\n        with mr_job.make_runner() as runner:\n            runner.run()\n            for line in runner.stream_output():\n                key, value = mr_job.parse_output_line(line)\n                yield value\n\n                \nclass TestMrS3LogParser(unittest.TestCase):\n\n    mr_job = None\n    mr_tests_util = None\n\n    RAW_LOG_LINE_INVALID = \\\n        \'00000fe9688b6e57f75bd2b7f7c1610689e8f01000000\' \\\n        \'00000388225bcc00000 \' \\\n        \'s3-storage [22/Jul/2013:21:03:27 +0000] \' \\\n        \'00.111.222.33 \' \\\n\n    RAW_LOG_LINE_VALID = \\\n        \'00000fe9688b6e57f75bd2b7f7c1610689e8f01000000\' \\\n        \'00000388225bcc00000 \' \\\n        \'s3-storage [22/Jul/2013:21:03:27 +0000] \' \\\n        \'00.111.222.33 \' \\\n        \'arn:aws:sts::000005646931:federated-user/user 00000AB825500000 \' \\\n        \'REST.HEAD.OBJECT user/file.pdf \' \\\n        \'"HEAD /user/file.pdf?versionId=00000XMHZJp6DjM9x500000\' \\\n        \'00000SDZk \' \\\n        \'HTTP/1.1" 200 - - 4000272 18 - "-" \' \\\n        \'"Boto/2.5.1 (darwin) USER-AGENT/1.0.14.0" \' \\\n        \'00000XMHZJp6DjM9x5JVEAMo8MG00000\'\n\n    DATE_TIME_ZONE_INVALID = "AB/Jul/2013:21:04:17 +0000"\n    DATE_TIME_ZONE_VALID = "22/Jul/2013:21:04:17 +0000"\n    DATE_VALID = "2013-07-22"\n    DATE_TIME_VALID = "2013-07-22 21:04:17"\n    TIME_ZONE_VALID = "+0000"\n\n    def __init__(self, *args, **kwargs):\n        super(TestMrS3LogParser, self).__init__(*args, **kwargs)\n        self.mr_job = MrS3LogParser([\'-r\', \'inline\', \'--no-conf\', \'-\'])\n        self.mr_tests_util = MrTestsUtil()\n\n    def test_invalid_log_lines(self):\n        stdin = StringIO(self.RAW_LOG_LINE_INVALID)\n\n        for result in self.mr_tests_util.run_mr_sandbox(self.mr_job, stdin):\n            self.assertEqual(result.find("Error"), 0)\n\n    def test_valid_log_lines(self):\n        stdin = StringIO(self.RAW_LOG_LINE_VALID)\n\n        for result in self.mr_tests_util.run_mr_sandbox(self.mr_job, stdin):\n            self.assertEqual(result.find("Error"), -1)\n\n    def test_clean_date_time_zone(self):\n        date, date_time, time_zone_parsed = \\\n            self.mr_job.clean_date_time_zone(self.DATE_TIME_ZONE_VALID)\n        self.assertEqual(date, self.DATE_VALID)\n        self.assertEqual(date_time, self.DATE_TIME_VALID)\n        self.assertEqual(time_zone_parsed, self.TIME_ZONE_VALID)\n\n        # Use a lambda to delay the calling of clean_date_time_zone so that\n        # assertRaises has enough time to handle it properly\n        self.assertRaises(ValueError,\n            lambda: self.mr_job.clean_date_time_zone(\n                self.DATE_TIME_ZONE_INVALID))\n\nif __name__ == \'__main__\':\n    unittest.main()\n')


# ## Running S3 Logs Unit Test

# Run the mrjob test:

# In[ ]:


# get_ipython().system('python test_mr_s3_log_parser.py -v')


# ## Sample Config File

# In[ ]:


runners:
  emr:
    aws_access_key_id: __ACCESS_KEY__
    aws_secret_access_key: __SECRET_ACCESS_KEY__
    aws_region: us-east-1
    ec2_key_pair: EMR
    ec2_key_pair_file: ~/.ssh/EMR.pem
    ssh_tunnel_to_job_tracker: true
    ec2_master_instance_type: m3.xlarge
    ec2_instance_type: m3.xlarge
    num_ec2_instances: 5
    s3_scratch_uri: s3://bucket/tmp/
    s3_log_uri: s3://bucket/tmp/logs/
    enable_emr_debugging: True
    bootstrap:
    - sudo apt-get install -y python-pip
    - sudo pip install --upgrade simplejson

