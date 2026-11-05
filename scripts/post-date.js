const siteTimezone = hexo.config.timezone || 'UTC';

hexo.extend.helper.register('post_date', (date, includeTime = false) => {
  const format = includeTime
    ? `${hexo.config.date_format} ${hexo.config.time_format}`
    : hexo.config.date_format;

  return date.tz(siteTimezone).format(format);
});
