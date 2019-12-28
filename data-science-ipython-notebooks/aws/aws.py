#!/usr/bin/env python
# coding: utf-8

# This notebook was prepared by [Donne Martin](http://donnemartin.com). Source and license info is on [GitHub](https://github.com/donnemartin/data-science-ipython-notebooks).

# # Amazon Web Services (AWS)
# 
# * SSH to EC2
# * Boto
# * S3cmd
# * s3-parallel-put
# * S3DistCp
# * Redshift
# * Kinesis
# * Lambda

# <h2 id="ssh-to-ec2">SSH to EC2</h2>

# Connect to an Ubuntu EC2 instance through SSH with the given key:

# In[ ]:


# get_ipython().system('ssh -i key.pem ubuntu@ipaddress')


# Connect to an Amazon Linux EC2 instance through SSH with the given key:

# In[ ]:


# get_ipython().system('ssh -i key.pem ec2-user@ipaddress')


# ## Boto
# 
# [Boto](https://github.com/boto/boto) is the official AWS SDK for Python.
# 
# Install Boto:

# In[ ]:


# get_ipython().system('pip install Boto')


# Configure boto by creating a ~/.boto file with the following:

# In[ ]:


aws_access_key_id = YOURACCESSKEY
aws_secret_access_key = YOURSECRETKEY


# Work with S3:

# In[ ]:


import boto
s3 = boto.connect_s3()


# Work with EC2:

# In[ ]:


import boto.ec2
ec2 = boto.ec2.connect_to_region('us-east-1')


# Create a bucket and put an object in that bucket:

# In[ ]:


import boto
import time
s3 = boto.connect_s3()

# Create a new bucket. Buckets must have a globally unique name (not just
# unique to your account).
bucket = s3.create_bucket('boto-demo-%s' % int(time.time()))

# Create a new key/value pair.
key = bucket.new_key('mykey')
key.set_contents_from_string("Hello World!")

# Sleep to ensure the data is eventually there.
# This is often referred to as "S3 eventual consistency".
time.sleep(2)

# Retrieve the contents of ``mykey``.
print key.get_contents_as_string()

# Delete the key.
key.delete()

# Delete the bucket.
bucket.delete()


# Each service supports a different set of commands. Refer to the following for more details:
# * [AWS Docs](https://aws.amazon.com/documentation/)
# * [Boto Docs](http://boto.readthedocs.org/en/latest/index.html)

# <h2 id="s3cmd">S3cmd</h2>
# 
# Before I discovered [S3cmd](http://s3tools.org/s3cmd), I had been using the [S3 console](http://aws.amazon.com/console/) to do basic operations and [boto](https://boto.readthedocs.org/en/latest/) to do more of the heavy lifting.  However, sometimes I just want to hack away at a command line to do my work.
# 
# I've found S3cmd to be a great command line tool for interacting with S3 on AWS.  S3cmd is written in Python, is open source, and is free even for commercial use.  It offers more advanced features than those found in the [AWS CLI](http://aws.amazon.com/cli/).

# Install s3cmd:

# In[ ]:


# get_ipython().system('sudo apt-get install s3cmd')


# Running the following command will prompt you to enter your AWS access and AWS secret keys. To follow security best practices, make sure you are using an IAM account as opposed to using the root account.
# 
# I also suggest enabling GPG encryption which will encrypt your data at rest, and enabling HTTPS to encrypt your data in transit.  Note this might impact performance.

# In[ ]:


# get_ipython().system('s3cmd --configure')


# Frequently used S3cmds:

# In[ ]:


# List all buckets
# get_ipython().system('s3cmd ls')

# List the contents of the bucket
# get_ipython().system('s3cmd ls s3://my-bucket-name')

# Upload a file into the bucket (private)
# get_ipython().system('s3cmd put myfile.txt s3://my-bucket-name/myfile.txt')

# Upload a file into the bucket (public)
# get_ipython().system('s3cmd put --acl-public --guess-mime-type myfile.txt s3://my-bucket-name/myfile.txt')

# Recursively upload a directory to s3
# get_ipython().system('s3cmd put --recursive my-local-folder-path/ s3://my-bucket-name/mydir/')

# Download a file
# get_ipython().system('s3cmd get s3://my-bucket-name/myfile.txt myfile.txt')

# Recursively download files that start with myfile
# get_ipython().system('s3cmd --recursive get s3://my-bucket-name/myfile')

# Delete a file
# get_ipython().system('s3cmd del s3://my-bucket-name/myfile.txt')

# Delete a bucket
# get_ipython().system('s3cmd del --recursive s3://my-bucket-name/')

# Create a bucket
# get_ipython().system('s3cmd mb s3://my-bucket-name')

# List bucket disk usage (human readable)
# get_ipython().system('s3cmd du -H s3://my-bucket-name/')

# Sync local (source) to s3 bucket (destination)
# get_ipython().system('s3cmd sync my-local-folder-path/ s3://my-bucket-name/')

# Sync s3 bucket (source) to local (destination)
# get_ipython().system('s3cmd sync s3://my-bucket-name/ my-local-folder-path/')

# Do a dry-run (do not perform actual sync, but get information about what would happen)
# get_ipython().system('s3cmd --dry-run sync s3://my-bucket-name/ my-local-folder-path/')

# Apply a standard shell wildcard include to sync s3 bucket (source) to local (destination)
# get_ipython().system("s3cmd --include '2014-05-01*' sync s3://my-bucket-name/ my-local-folder-path/")


# <h2 id="s3-parallel-put">s3-parallel-put</h2>
# 
# [s3-parallel-put](https://github.com/twpayne/s3-parallel-put.git) is a great tool for uploading multiple files to S3 in parallel.

# Install package dependencies:

# In[ ]:


# get_ipython().system('sudo apt-get install boto')
# get_ipython().system('sudo apt-get install git')


# Clone the s3-parallel-put repo:

# In[ ]:


# get_ipython().system('git clone https://github.com/twpayne/s3-parallel-put.git')


# Setup AWS keys for s3-parallel-put:

# In[ ]:


# get_ipython().system('export AWS_ACCESS_KEY_ID=XXX')
# get_ipython().system('export AWS_SECRET_ACCESS_KEY=XXX')


# Sample usage:

# In[ ]:


# get_ipython().system('s3-parallel-put --bucket=bucket --prefix=PREFIX SOURCE')


# Dry run of putting files in the current directory on S3 with the given S3 prefix, do not check first if they exist:

# In[ ]:


# get_ipython().system('s3-parallel-put --bucket=bucket --host=s3.amazonaws.com --put=stupid --dry-run --prefix=prefix/ ./')


# <h2 id="s3distcp">S3DistCp</h2>
# 
# [S3DistCp](http://docs.aws.amazon.com/ElasticMapReduce/latest/DeveloperGuide/UsingEMR_s3distcp.html) is an extension of DistCp that is optimized to work with Amazon S3.  S3DistCp is useful for combining smaller files and aggregate them together, taking in a pattern and target file to combine smaller input files to larger ones.  S3DistCp can also be used to transfer large volumes of data from S3 to your Hadoop cluster.

# To run S3DistCp with the EMR command line, ensure you are using the proper version of Ruby:

# In[ ]:


# get_ipython().system('rvm --default ruby-1.8.7-p374')


# The EMR command line below executes the following:
# * Create a master node and slave nodes of type m1.small
# * Runs S3DistCp on the source bucket location and concatenates files that match the date regular expression, resulting in files that are roughly 1024 MB or 1 GB
# * Places the results in the destination bucket

# In[ ]:


# get_ipython().system('./elastic-mapreduce --create --instance-group master --instance-count 1 --instance-type m1.small --instance-group core --instance-count 4 --instance-type m1.small --jar /home/hadoop/lib/emr-s3distcp-1.0.jar --args "--src,s3://my-bucket-source/,--groupBy,.*([0-9]{4}-01).*,--dest,s3://my-bucket-dest/,--targetSize,1024"')


# For further optimization, compression can be helpful to save on AWS storage and bandwidth costs, to speed up the S3 to/from EMR transfer, and to reduce disk I/O. Note that compressed files are not easy to split for Hadoop. For example, Hadoop uses a single mapper per GZIP file, as it does not know about file boundaries.
# 
# What type of compression should you use?
# 
# * Time sensitive job: Snappy or LZO
# * Large amounts of data: GZIP
# * General purpose: GZIP, as it’s supported by most platforms
# 
# You can specify the compression codec (gzip, lzo, snappy, or none) to use for copied files with S3DistCp with –outputCodec. If no value is specified, files are copied with no compression change. The code below sets the compression to lzo:

# In[ ]:


--outputCodec,lzo


# <h2 id="redshift">Redshift</h2>

# Copy values from the given S3 location containing CSV files to a Redshift cluster:

# In[ ]:


copy table_name from 's3://source/part'
credentials 'aws_access_key_id=XXX;aws_secret_access_key=XXX'
csv;


# Copy values from the given location containing TSV files to a Redshift cluster:

# In[ ]:


copy table_name from 's3://source/part'
credentials 'aws_access_key_id=XXX;aws_secret_access_key=XXX'
csv delimiter '\t';


# View Redshift errors:

# In[ ]:


select * from stl_load_errors;


# Vacuum Redshift in full:

# In[ ]:


VACUUM FULL;


# Analyze the compression of a table:

# In[ ]:


analyze compression table_name;


# Cancel the query with the specified id:

# In[ ]:


cancel 18764;


# The CANCEL command will not abort a transaction. To abort or roll back a transaction, you must use the ABORT or ROLLBACK command. To cancel a query associated with a transaction, first cancel the query then abort the transaction.
# 
# If the query that you canceled is associated with a transaction, use the ABORT or ROLLBACK. command to cancel the transaction and discard any changes made to the data:

# In[ ]:


abort;


# Reference table creation and setup:

# ![alt text](http://docs.aws.amazon.com/redshift/latest/dg/images/tutorial-optimize-tables-ssb-data-model.png)

# In[ ]:


CREATE TABLE part (
  p_partkey             integer         not null        sortkey distkey,
  p_name                varchar(22)     not null,
  p_mfgr                varchar(6)      not null,
  p_category            varchar(7)      not null,
  p_brand1              varchar(9)      not null,
  p_color               varchar(11)     not null,
  p_type                varchar(25)     not null,
  p_size                integer         not null,
  p_container           varchar(10)     not null
);

CREATE TABLE supplier (
  s_suppkey             integer        not null sortkey,
  s_name                varchar(25)    not null,
  s_address             varchar(25)    not null,
  s_city                varchar(10)    not null,
  s_nation              varchar(15)    not null,
  s_region              varchar(12)    not null,
  s_phone               varchar(15)    not null)
diststyle all;

CREATE TABLE customer (
  c_custkey             integer        not null sortkey,
  c_name                varchar(25)    not null,
  c_address             varchar(25)    not null,
  c_city                varchar(10)    not null,
  c_nation              varchar(15)    not null,
  c_region              varchar(12)    not null,
  c_phone               varchar(15)    not null,
  c_mktsegment          varchar(10)    not null)
diststyle all;

CREATE TABLE dwdate (
  d_datekey            integer       not null sortkey,
  d_date               varchar(19)   not null,
  d_dayofweek          varchar(10)   not null,
  d_month              varchar(10)   not null,
  d_year               integer       not null,
  d_yearmonthnum       integer       not null,
  d_yearmonth          varchar(8)    not null,
  d_daynuminweek       integer       not null,
  d_daynuminmonth      integer       not null,
  d_daynuminyear       integer       not null,
  d_monthnuminyear     integer       not null,
  d_weeknuminyear      integer       not null,
  d_sellingseason      varchar(13)   not null,
  d_lastdayinweekfl    varchar(1)    not null,
  d_lastdayinmonthfl   varchar(1)    not null,
  d_holidayfl          varchar(1)    not null,
  d_weekdayfl          varchar(1)    not null)
diststyle all;

CREATE TABLE lineorder (
  lo_orderkey               integer     not null,
  lo_linenumber         integer         not null,
  lo_custkey            integer         not null,
  lo_partkey            integer         not null distkey,
  lo_suppkey            integer         not null,
  lo_orderdate          integer         not null sortkey,
  lo_orderpriority      varchar(15)     not null,
  lo_shippriority       varchar(1)      not null,
  lo_quantity           integer         not null,
  lo_extendedprice      integer         not null,
  lo_ordertotalprice    integer         not null,
  lo_discount           integer         not null,
  lo_revenue            integer         not null,
  lo_supplycost         integer         not null,
  lo_tax                integer         not null,
  lo_commitdate         integer         not null,
  lo_shipmode           varchar(10)     not null
);


# | Table name | Sort Key     | Distribution Style |
# |------------|--------------|--------------------|
# | LINEORDER  | lo_orderdate | lo_partkey         |
# | PART       | p_partkey    | p_partkey          |
# | CUSTOMER   | c_custkey    | ALL                |
# | SUPPLIER   | s_suppkey    | ALL                |
# | DWDATE     | d_datekey    | ALL                |

# [Sort Keys](http://docs.aws.amazon.com/redshift/latest/dg/tutorial-tuning-tables-sort-keys.html)
# 
# When you create a table, you can specify one or more columns as the sort key. Amazon Redshift stores your data on disk in sorted order according to the sort key. How your data is sorted has an important effect on disk I/O, columnar compression, and query performance.
# 
# Choose sort keys for based on these best practices:
# 
# If recent data is queried most frequently, specify the timestamp column as the leading column for the sort key.
# 
# If you do frequent range filtering or equality filtering on one column, specify that column as the sort key.
# 
# If you frequently join a (dimension) table, specify the join column as the sort key.

# [Distribution Styles](http://docs.aws.amazon.com/redshift/latest/dg/c_choosing_dist_sort.html)
# 
# When you create a table, you designate one of three distribution styles: KEY, ALL, or EVEN.
# 
# **KEY distribution**
# 
# The rows are distributed according to the values in one column. The leader node will attempt to place matching values on the same node slice. If you distribute a pair of tables on the joining keys, the leader node collocates the rows on the slices according to the values in the joining columns so that matching values from the common columns are physically stored together.
# 
# **ALL distribution**
# 
# A copy of the entire table is distributed to every node. Where EVEN distribution or KEY distribution place only a portion of a table's rows on each node, ALL distribution ensures that every row is collocated for every join that the table participates in.
# 
# **EVEN distribution**
# 
# The rows are distributed across the slices in a round-robin fashion, regardless of the values in any particular column. EVEN distribution is appropriate when a table does not participate in joins or when there is not a clear choice between KEY distribution and ALL distribution. EVEN distribution is the default distribution style.

# <h2 id="kinesis">Kinesis</h2>

# Create a stream:

# In[ ]:


# get_ipython().system('aws kinesis create-stream --stream-name Foo --shard-count 1 --profile adminuser')


# List all streams:

# In[ ]:


# get_ipython().system('aws kinesis list-streams --profile adminuser')


# Get info about the stream:

# In[ ]:


# get_ipython().system('aws kinesis describe-stream --stream-name Foo --profile adminuser')


# Put a record to the stream:

# In[ ]:


# get_ipython().system('aws kinesis put-record --stream-name Foo --data "SGVsbG8sIHRoaXMgaXMgYSB0ZXN0IDEyMy4=" --partition-key shardId-000000000000 --region us-east-1 --profile adminuser')


# Get records from a given shard:

# In[ ]:


# get_ipython().system("SHARD_ITERATOR=$(aws kinesis get-shard-iterator --shard-id shardId-000000000000 --shard-iterator-type TRIM_HORIZON --stream-name Foo --query 'ShardIterator' --profile adminuser)")
aws kinesis get-records --shard-iterator $SHARD_ITERATOR


# Delete a stream:

# In[ ]:


# get_ipython().system('aws kinesis delete-stream --stream-name Foo --profile adminuser')


# <h2 id="lambda">Lambda</h2>

# List lambda functions:

# In[ ]:


# get_ipython().system('aws lambda list-functions     --region us-east-1     --max-items 10')


# Upload a lambda function:

# In[ ]:


# get_ipython().system('aws lambda upload-function     --region us-east-1     --function-name foo     --function-zip file-path/foo.zip     --role IAM-role-ARN     --mode event     --handler foo.handler     --runtime nodejs     --debug')


# Invoke a lambda function:

# In[ ]:


# get_ipython().system('aws lambda  invoke-async     --function-name foo     --region us-east-1     --invoke-args foo.txt     --debug')


# Update a function:

# In[ ]:


# get_ipython().system('aws lambda update-function-configuration    --function-name foo     --region us-east-1    --timeout timeout-in-seconds ')


# Return metadata for a specific function:

# In[ ]:


# get_ipython().system('aws lambda get-function-configuration     --function-name foo     --region us-east-1     --debug')


# Return metadata for a specific function along with a presigned URL that you can use to download the function's .zip file that you uploaded:

# In[ ]:


# get_ipython().system('aws lambda get-function     --function-name foo     --region us-east-1     --debug')


# Add an event source:

# In[ ]:


# get_ipython().system('aws lambda add-event-source     --region us-east-1     --function-name ProcessKinesisRecords     --role invocation-role-arn      --event-source kinesis-stream-arn     --batch-size 100')


# Add permissions:

# In[ ]:


# get_ipython().system('aws lambda add-permission     --function-name CreateThumbnail     --region us-west-2     --statement-id some-unique-id     --action "lambda:InvokeFunction"     --principal s3.amazonaws.com     --source-arn arn:aws:s3:::sourcebucket     --source-account bucket-owner-account-id')


# Check policy permissions:

# In[ ]:


# get_ipython().system('aws lambda get-policy     --function-name function-name')


# Delete a lambda function:

# In[ ]:


# get_ipython().system('aws lambda delete-function     --function-name foo     --region us-east-1     --debug')

