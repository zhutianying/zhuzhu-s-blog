hexo.extend.generator.register('latest-post', function (locals) {
  const post = locals.posts.sort('-date').first();
  if (!post) return [];

  return {
    path: '',
    layout: 'home',
    data: {
      post,
      title: post.title,
      description: post.description,
      date: post.date,
      cover: post.cover,
    },
  };
});
