/**
 * Created by Xiaotao.Nie on 09/04/2018.
 * All right reserved
 * IF you have any question please email onlythen@yeah.net
 */

// Global functions and listeners
window.onresize = () => {
    // when window resize , we show remove some class that me be added
    // often for debug
    if(window.document.documentElement.clientWidth > 680){
        const aboutContent = document.getElementById('nav-content')
        if (aboutContent) {
            aboutContent.classList.remove('hide-block')
            aboutContent.classList.remove('show-block');
        }
    }
    // if(window.isPost){
        // reLayout()
    // }

    reHeightToc();
};

// Nav switch function on mobile
/*****************************************************************************/
const navToggle = document.getElementById('site-nav-toggle');
navToggle.addEventListener('click', () => {
    const aboutContent = document.getElementById('nav-content')
    if (!aboutContent) return;

    if (!aboutContent.classList.contains('show-block')) {
        aboutContent.classList.add('show-block');
        aboutContent.classList.remove('hide-block')
        navToggle.setAttribute('aria-expanded', 'true');
    } else {
        aboutContent.classList.add('hide-block')
        aboutContent.classList.remove('show-block');
        navToggle.setAttribute('aria-expanded', 'false');
    }
})


// global search
/*****************************************************************************/

const searchButton = document.getElementById('search')
const searchField = document.getElementById('search-field')
const searchInput = document.getElementById('search-input')
const searchResultContainer = document.getElementById('search-result-container')
const escSearch = document.getElementById('esc-search')
const bgSearch = document.getElementById('search-bg')
const beginSearch = document.getElementById('begin-search')

searchField.addEventListener('wheel',(e) => {
    // e.preventDefault()
    e.stopPropagation()
    return false
}, false)

let searchJson;
let searchLoading;
const caseSensitive = false;
const searchWindowSize = 100;
const maximumSnippetsPerResult = 3;

searchButton.addEventListener('click', (event) => {
    event.preventDefault();
    search()
});

searchButton.addEventListener('keydown', (event) => {
    if (event.key !== 'Enter' && event.key !== ' ') return;
    event.preventDefault();
    search();
});

escSearch.addEventListener('click',() => {
    hideSearchField()
})

bgSearch.addEventListener('click',() => {
    hideSearchField()
})

beginSearch.addEventListener('click',() => {
    searchFromInput()
})

searchInput.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') searchFromInput();
});

function toggleSeachField(){
    if (!searchField.classList.contains('show-flex-fade')) {
        showSearchField()
    } else {
        hideSearchField()
    }
}

function showSearchField() {
    searchInput.focus()
    searchField.classList.add('show-flex-fade');
    searchField.classList.remove('hide-flex-fade');
}

function hideSearchField(){
    searchField.classList.add('hide-flex-fade');
    searchField.classList.remove('show-flex-fade');
}

function isSearchVisible() {
    return searchField.classList.contains('show-flex-fade');
}

function clearSearchResults() {
    searchResultContainer.replaceChildren();
}

function showSearchMessage(message) {
    clearSearchResults();
    const result = document.createElement('div');
    result.className = 'no-search-result';
    result.textContent = message;
    searchResultContainer.appendChild(result);
}

function appendHighlightedText(container, text, keyword) {
    const comparisonText = caseSensitive ? text : text.toLowerCase();
    const comparisonKeyword = caseSensitive ? keyword : keyword.toLowerCase();
    let cursor = 0;
    let matchStart = comparisonText.indexOf(comparisonKeyword, cursor);

    while (matchStart !== -1) {
        container.append(document.createTextNode(text.slice(cursor, matchStart)));
        const highlight = document.createElement('span');
        highlight.className = 'red';
        highlight.textContent = text.slice(matchStart, matchStart + keyword.length);
        container.append(highlight);
        cursor = matchStart + keyword.length;
        matchStart = comparisonText.indexOf(comparisonKeyword, cursor);
    }

    container.append(document.createTextNode(text.slice(cursor)));
}

function createSnippet(content, matchStart, keyword) {
    const start = Math.max(0, matchStart - searchWindowSize / 2);
    const end = Math.min(content.length, start + searchWindowSize);
    const snippet = document.createDocumentFragment();
    if (start > 0) snippet.append(document.createTextNode('…'));
    appendHighlightedText(snippet, content.slice(start, end), keyword);
    if (end < content.length) snippet.append(document.createTextNode('…'));
    return snippet;
}

function searchFromKeyWord(keyword = "") {
    const normalizedKeyword = caseSensitive ? keyword : keyword.toLowerCase();
    const results = [];

    searchJson.forEach((item) => {
        if (!item.title || !item.content) return;

        const title = String(item.title);
        const content = String(item.content).trim().replace(/<[^>]+>/g, '').replace(/[`#\n]/g, '');
        const searchableTitle = caseSensitive ? title : title.toLowerCase();
        const searchableContent = caseSensitive ? content : content.toLowerCase();
        const snippets = [];
        let matchStart = searchableContent.indexOf(normalizedKeyword);

        while (matchStart !== -1 && snippets.length < maximumSnippetsPerResult) {
            snippets.push(matchStart);
            matchStart = searchableContent.indexOf(normalizedKeyword, matchStart + normalizedKeyword.length);
        }

        if (searchableTitle.includes(normalizedKeyword) || snippets.length) {
            results.push({ title, content, url: item.url, snippets });
        }
    });

    if (!results.length) {
        showSearchMessage('没有找到匹配的文章。');
        return;
    }

    const searchFragment = document.createElement('ul');
    results.forEach((item) => {
        const searchItem = document.createElement('li');
        const searchTitle = document.createElement('a');
        searchTitle.href = item.url;
        appendHighlightedText(searchTitle, item.title, keyword);
        searchItem.appendChild(searchTitle);

        if (item.snippets.length) {
            const snippets = document.createElement('ul');
            item.snippets.forEach((matchStart) => {
                const snippet = document.createElement('li');
                snippet.append(createSnippet(item.content, matchStart, keyword));
                snippets.appendChild(snippet);
            });
            searchItem.appendChild(snippets);
        }

        searchFragment.appendChild(searchItem);
    });

    clearSearchResults();
    searchResultContainer.appendChild(searchFragment);
}

async function loadSearchIndex() {
    if (searchJson) return searchJson;
    if (searchLoading) return searchLoading;

    let searchPath = window.hexo_search_path || 'search.json';
    if (!/json$/i.test(searchPath)) {
        throw new Error('本站搜索仅支持 JSON 索引。');
    }

    searchLoading = fetch(window.hexo_root + searchPath)
        .then((response) => {
            if (!response.ok) throw new Error(`搜索索引加载失败（${response.status}）。`);
            return response.json();
        })
        .then((data) => {
            if (!Array.isArray(data)) throw new Error('搜索索引格式无效。');
            searchJson = data;
            return searchJson;
        })
        .finally(() => {
            searchLoading = undefined;
        });

    return searchLoading;
}

async function searchFromInput() {
    const keyword = searchInput.value.trim();
    if (!keyword) return;

    try {
        if (!searchJson) showSearchMessage('正在加载搜索索引…');
        await loadSearchIndex();
        searchFromKeyWord(keyword);
    } catch (error) {
        showSearchMessage(error.message || '搜索暂时不可用。');
    }
}

function search() {
    toggleSeachField();
    if (!isSearchVisible()) return;
    loadSearchIndex().catch((error) => showSearchMessage(error.message || '搜索暂时不可用。'));
}

document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && isSearchVisible()) hideSearchField();
});

// directory function in post pages
/*****************************************************************************/
function getDistanceOfLeft(obj) {
    let left = 0;
    let top = 0;
    while (obj) {
        left += obj.offsetLeft;
        top += obj.offsetTop;
        obj = obj.offsetParent;
    }
    return {
        left:left,
        top:top
    };
}

var toc = document.getElementById('toc')

var tocToTop = getDistanceOfLeft(toc).top;

function reHeightToc(){
    if(toc) { // resize toc height
        toc.style.maxHeight = ( document.documentElement.clientHeight - 10 ) + 'px';
        toc.style.overflowY = 'scroll';
    }
}

reHeightToc();

if(window.isPost){
    var result = []

    var nameSet = new Set();

    if(!toc || !toc.children || !toc.children[0]){
        // do nothing
    }
    else {
        if (toc.children[0].nodeName === "OL") {
            let ol = Array.from(toc.children[0].children)

            function getArrayFromOl(ol) {
                let result = []

                // let escape = function (item) {
                //     return item.replace(/<[^>]+>/g, "").replace(/^\s\s*/, '').replace(/\s\s*$/, '').replace(/[\. _]/g, '-')
                // }

                ol.forEach((item) => {
                    if (item.children.length === 1) {
                        // TODO: need change
                        let value = item.children[0].getAttribute('href').replace(/^#/,"")
                        result.push({
                            value: [value],
                            dom: item
                        })
                        nameSet.add(value)
                    }
                    else {
                        let concatArray = getArrayFromOl(Array.from(item.children[1].children))
                        nameSet.add(item.children[0].getAttribute('href').replace(/^#/,""))
                        result.push({
                            value: [item.children[0].getAttribute('href').replace(/^#/,"")].concat(concatArray.reduce((p, n) => {
                                p = p.concat(n.value)
                                return p;
                            }, [])),
                            dom: item
                        })
                        result = result.concat(concatArray)
                    }
                })
                return result
            }

            result = getArrayFromOl(ol)
        }

        var nameArray = Array.from(nameSet)

        function reLayout() {
            let scrollToTop = document.documentElement.scrollTop || window.pageYOffset // Safari is special
            if(tocToTop === 0) {
                // Fix bug that when resize window the toc layout may be wrong
                toc = document.getElementById('toc')
                toc.classList.remove('toc-fixed')
                tocToTop = getDistanceOfLeft(toc).top;
            }
            if (tocToTop <= scrollToTop + 10) {
                if (!toc.classList.contains('toc-fixed'))
                    toc.classList.add('toc-fixed')
            } else {
                if (toc.classList.contains('toc-fixed'))
                    toc.classList.remove('toc-fixed')
            }

            let minTop = 9999;
            let minTopsValue = ""

            for (let item of nameArray) {
                item = decodeURIComponent(item);
                let dom = document.getElementById(item) || document.getElementById(item.replace(/\s/g, ''))
                if (!dom) {
                    console.log('dom is null')
                    continue
                }
                let toTop = getDistanceOfLeft(dom).top - scrollToTop;

                if (Math.abs(toTop) < minTop) {
                    minTop = Math.abs(toTop)
                    minTopsValue = item
                }
                // console.log(minTopsValue, minTop)
            }

            if (minTopsValue) {
                for (let item of result) {
                    if (item.value.indexOf(encodeURIComponent(minTopsValue)) !== -1) {
                        item.dom.classList.add("active")
                    } else {
                        item.dom.classList.remove("active")
                    }
                }
            }
        }

        reLayout()

        window.addEventListener('scroll', function(e) {
            reLayout()
            // let tocDom = document.querySelector('#toc')
            // window.scrollY < 550 ? tocDom.classList.remove('toc-fixed') : tocDom.classList.add('toc-fixed')
        })
    }
}


// donate
/*****************************************************************************/
const donateButton = document.getElementById('donate-button')
const donateImgContainer = document.getElementById('donate-img-container')
const donateImg = document.getElementById('donate-img')

if(donateButton) {
    donateButton.addEventListener('click', () => {
        if (donateImgContainer.classList.contains('hide')) {
            donateImgContainer.classList.remove('hide')
        } else {
            donateImgContainer.classList.add('hide')
        }
    })

    donateImg.src = donateImg.dataset.src
}

